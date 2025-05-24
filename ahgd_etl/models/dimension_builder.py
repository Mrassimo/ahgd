"""
Enhanced Dimension Builder with automatic unknown member generation.

This module builds dimension tables with integrated data quality fixes,
including automatic addition of unknown members.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import polars as pl

from ahgd_etl.config import settings
from ahgd_etl.core.fix_manager import FixManager
from ahgd_etl.loaders.parquet import ParquetLoader


class DimensionBuilder:
    """Builds dimension tables with automatic unknown members."""
    
    def __init__(self, output_dir: Path, fix_manager: Optional[FixManager] = None):
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        self.fix_manager = fix_manager or FixManager()
        self.loader = ParquetLoader(self.output_dir)
        
    def build_health_condition_dimension(self) -> bool:
        """Build health condition dimension with unknown member."""
        try:
            self.logger.info("Building health condition dimension...")
            
            # Get health conditions from configuration
            conditions = settings.dimensions.health_conditions
            
            # Create dimension records
            records = []
            for code, name in conditions.items():
                # Determine category based on condition
                if "mental" in name.lower():
                    category = "Mental Health"
                elif "heart" in name.lower() or "stroke" in name.lower():
                    category = "Cardiovascular"
                elif "diabetes" in name.lower():
                    category = "Metabolic"
                elif "cancer" in name.lower():
                    category = "Oncology"
                elif "dementia" in name.lower() or "alzheimer" in name.lower():
                    category = "Neurological"
                else:
                    category = "Other"
                    
                records.append({
                    "condition_sk": self.fix_manager.generate_surrogate_key("health_condition", code),
                    "condition_code": code,
                    "condition_name": name,
                    "condition_category": category,
                    "is_unknown": False,
                    "etl_processed_at": datetime.now()
                })
            
            # Create dataframe
            df = pl.DataFrame(records)
            
            # Add unknown member
            df = self.fix_manager.add_unknown_dimension_member(
                df,
                "dim_health_condition",
                "condition_sk",
                {
                    "condition_code": "UNKNOWN",
                    "condition_name": "Unknown Condition",
                    "condition_category": "Unknown"
                }
            )
            
            # Enforce schema
            schema = settings.schemas.dimensions.get("dim_health_condition", {})
            if schema:
                df = self.fix_manager.enforce_schema(df, schema, "dim_health_condition")
            
            # Save
            output_path = self.output_dir / "dim_health_condition.parquet"
            self.loader.save(df, output_path)
            
            self.logger.info(f"Health condition dimension created with {len(df)} records")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to build health condition dimension: {str(e)}", exc_info=True)
            return False
    
    def build_demographic_dimension(self) -> bool:
        """Build demographic dimension with unknown member."""
        try:
            self.logger.info("Building demographic dimension...")
            
            # Get demographic categories from configuration
            demographics = settings.dimensions.demographic_categories
            
            # Create dimension records
            records = []
            for category in demographics:
                age_group = category.get("age_group", "")
                sex = category.get("sex", "")
                
                records.append({
                    "demographic_sk": self.fix_manager.generate_surrogate_key(
                        "demographic", age_group, sex
                    ),
                    "age_group": age_group,
                    "sex": sex,
                    "is_unknown": False,
                    "etl_processed_at": datetime.now()
                })
            
            # Create dataframe
            df = pl.DataFrame(records)
            
            # Add unknown member
            df = self.fix_manager.add_unknown_dimension_member(
                df,
                "dim_demographic",
                "demographic_sk",
                {
                    "age_group": "Unknown",
                    "sex": "U"
                }
            )
            
            # Enforce schema
            schema = settings.schemas.dimensions.get("dim_demographic", {})
            if schema:
                df = self.fix_manager.enforce_schema(df, schema, "dim_demographic")
            
            # Save
            output_path = self.output_dir / "dim_demographic.parquet"
            self.loader.save(df, output_path)
            
            self.logger.info(f"Demographic dimension created with {len(df)} records")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to build demographic dimension: {str(e)}", exc_info=True)
            return False
    
    def build_person_characteristic_dimension(self) -> bool:
        """Build person characteristic dimension with unknown member."""
        try:
            self.logger.info("Building person characteristic dimension...")
            
            # Get person characteristics from configuration
            characteristics = settings.dimensions.person_characteristics
            
            # Create dimension records
            records = []
            for char_type, values in characteristics.items():
                for code, name in values.items():
                    # Determine category
                    if char_type == "country_of_birth":
                        category = "Country of Birth"
                    elif char_type == "labour_force":
                        category = "Labour Force Status"
                    elif char_type == "income":
                        category = "Income Range"
                    elif char_type == "education":
                        category = "Education Level"
                    else:
                        category = "Other"
                        
                    records.append({
                        "characteristic_sk": self.fix_manager.generate_surrogate_key(
                            "person_characteristic", char_type, code
                        ),
                        "characteristic_type": char_type,
                        "characteristic_value": name,
                        "characteristic_code": code,
                        "characteristic_category": category,
                        "is_unknown": False,
                        "etl_processed_at": datetime.now()
                    })
            
            # Create dataframe
            df = pl.DataFrame(records)
            
            # Add unknown member
            df = self.fix_manager.add_unknown_dimension_member(
                df,
                "dim_person_characteristic",
                "characteristic_sk",
                {
                    "characteristic_type": "Unknown",
                    "characteristic_value": "Unknown",
                    "characteristic_code": "UNK",
                    "characteristic_category": "Unknown"
                }
            )
            
            # Enforce schema
            schema = settings.schemas.dimensions.get("dim_person_characteristic", {})
            if schema:
                df = self.fix_manager.enforce_schema(df, schema, "dim_person_characteristic")
            
            # Save
            output_path = self.output_dir / "dim_person_characteristic.parquet"
            self.loader.save(df, output_path)
            
            self.logger.info(f"Person characteristic dimension created with {len(df)} records")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to build person characteristic dimension: {str(e)}", exc_info=True)
            return False
    
    def build_all_dimensions(self) -> bool:
        """Build all dimension tables."""
        success = True
        
        # Build each dimension
        dimensions = [
            ("Health Condition", self.build_health_condition_dimension),
            ("Demographic", self.build_demographic_dimension),
            ("Person Characteristic", self.build_person_characteristic_dimension)
        ]
        
        for dim_name, builder_func in dimensions:
            self.logger.info(f"Building {dim_name} dimension...")
            if not builder_func():
                self.logger.error(f"Failed to build {dim_name} dimension")
                success = False
                
        return success
    
    def update_geo_dimension_unknown(self) -> bool:
        """Add unknown member to existing geo dimension."""
        try:
            geo_path = self.output_dir / "geo_dimension.parquet"
            if not geo_path.exists():
                self.logger.warning("Geo dimension not found, skipping unknown member addition")
                return True
                
            # Load existing dimension
            df = pl.read_parquet(geo_path)
            
            # Add unknown member
            df = self.fix_manager.add_unknown_dimension_member(
                df,
                "geo_dimension",
                "geo_sk"
            )
            
            # Save back
            df.write_parquet(geo_path)
            
            self.logger.info("Added unknown member to geo dimension")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update geo dimension: {str(e)}", exc_info=True)
            return False