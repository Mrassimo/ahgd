"""
Health Data Processor for Australian Medicare (MBS) and Pharmaceutical (PBS) data.

Processes real health utilisation data from Australian Government sources:
- Medicare Benefits Schedule (MBS) demographics and utilisation
- Pharmaceutical Benefits Scheme (PBS) prescriptions and costs
- Links health data with SA2 geographic areas and SEIFA socio-economic indices
- Enables population health analysis and risk assessment

Based on data.gov.au validated datasets.
"""

import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import polars as pl
from loguru import logger
from rich.console import Console

console = Console()

# MBS (Medicare Benefits Schedule) configuration
MBS_CONFIG = {
    "historical_filename": "MBS_Demographics_Historical_1993-2015.zip",
    "expected_records_historical": 1000000,  # Approximate record count
    "demographic_columns": ["age_group", "gender", "postcode", "state"],
    "service_columns": ["item_number", "service_category", "benefit_amount"],
    "temporal_columns": ["year", "quarter", "month"]
}

# PBS (Pharmaceutical Benefits Scheme) configuration
PBS_CONFIG = {
    "current_filename": "PBS_Item_Report_2016_Current.csv",
    "historical_filename": "PBS_Item_Historical_1992-2014.zip", 
    "expected_records_current": 500000,  # Approximate record count
    "prescription_columns": ["atc_code", "drug_name", "strength", "form"],
    "utilisation_columns": ["ddd_per_1000_pop_per_day", "prescription_count"],
    "cost_columns": ["cost_to_government", "patient_contribution"]
}

# Health data schema mappings
MBS_SCHEMA = {
    "year": pl.Int16,
    "quarter": pl.Int8, 
    "state": pl.Utf8,
    "postcode": pl.Utf8,
    "age_group": pl.Utf8,
    "gender": pl.Utf8,
    "item_number": pl.Utf8,
    "service_category": pl.Utf8,
    "services_count": pl.Int32,
    "benefit_amount": pl.Float64,
    "schedule_fee": pl.Float64
}

PBS_SCHEMA = {
    "year": pl.Int16,
    "month": pl.Int8,
    "state": pl.Utf8,
    "atc_code": pl.Utf8,
    "drug_name": pl.Utf8,
    "strength": pl.Utf8,
    "prescriptions": pl.Int32,
    "ddd_per_1000": pl.Float64,
    "cost_government": pl.Float64,
    "cost_patient": pl.Float64
}

# Australian health data classifications
HEALTH_CLASSIFICATIONS = {
    "age_groups": [
        "0-4", "5-14", "15-24", "25-34", "35-44", "45-54", 
        "55-64", "65-74", "75-84", "85+"
    ],
    "service_categories": [
        "Professional Attendances", "Diagnostic Procedures", "Therapeutic Procedures",
        "Oral and Maxillofacial", "Optometry", "Operations", "Pathology", "Radiology"
    ],
    "atc_categories": [
        "A - Alimentary Tract", "B - Blood", "C - Cardiovascular", 
        "D - Dermatological", "G - Genitourinary", "H - Hormones",
        "J - Anti-infectives", "L - Antineoplastic", "M - Musculoskeletal",
        "N - Nervous System", "P - Antiparasitic", "R - Respiratory",
        "S - Sensory Organs", "V - Various"
    ]
}


class HealthDataProcessor:
    """
    High-performance processor for Australian health utilisation data.
    
    Processes MBS and PBS data for population health analysis and risk modelling.
    """
    
    def __init__(self, data_dir: Union[str, Path] = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Health Data Processor initialized")
        logger.info(f"Raw data directory: {self.raw_dir}")
        logger.info(f"Processed data directory: {self.processed_dir}")
    
    def validate_health_file(self, file_path: Path, data_type: str = "mbs") -> bool:
        """
        Validate health data file structure.
        
        Args:
            file_path: Path to health data file
            data_type: Type of data ("mbs" or "pbs")
            
        Returns:
            True if file is valid, False otherwise
        """
        try:
            if not file_path.exists():
                logger.error(f"Health data file not found: {file_path}")
                return False
            
            # Check file size expectations
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            if data_type == "mbs":
                expected_size = 50  # MBS historical is ~50MB
                if file_size_mb < expected_size * 0.5 or file_size_mb > expected_size * 2:
                    logger.warning(f"Unexpected MBS file size: {file_size_mb:.1f}MB")
            elif data_type == "pbs":
                expected_size = 10 if file_path.suffix == ".csv" else 25
                if file_size_mb < expected_size * 0.5 or file_size_mb > expected_size * 2:
                    logger.warning(f"Unexpected PBS file size: {file_size_mb:.1f}MB")
            
            # Validate file format
            if file_path.suffix == ".zip":
                try:
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        file_list = zip_ref.namelist()
                        
                        # Should contain CSV files
                        csv_files = [f for f in file_list if f.endswith('.csv')]
                        if not csv_files:
                            logger.error("No CSV files found in ZIP archive")
                            return False
                        
                        logger.info(f"Found {len(csv_files)} CSV files in archive")
                        
                except zipfile.BadZipFile:
                    logger.error("Invalid ZIP file format")
                    return False
            
            elif file_path.suffix == ".csv":
                # Quick CSV validation - check first few lines
                try:
                    sample_df = pl.read_csv(file_path, n_rows=5)
                    if len(sample_df.columns) < 3:
                        logger.error("CSV file has too few columns")
                        return False
                        
                except Exception as e:
                    logger.error(f"Cannot read CSV file: {e}")
                    return False
            
            logger.info(f"✓ Health data file validation passed: {file_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Health data file validation failed: {e}")
            return False
    
    def extract_mbs_data(self, zip_path: Path) -> pl.DataFrame:
        """
        Extract MBS (Medicare) data from ZIP file.
        
        Args:
            zip_path: Path to MBS ZIP file
            
        Returns:
            Polars DataFrame with standardized MBS schema
        """
        if not self.validate_health_file(zip_path, "mbs"):
            raise ValueError(f"Invalid MBS file: {zip_path}")
        
        logger.info(f"Extracting MBS data from {zip_path.name}")
        
        try:
            # Extract ZIP to temporary directory
            extract_dir = self.raw_dir / "temp_mbs"
            extract_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Find CSV files
            csv_files = list(extract_dir.rglob("*.csv"))
            if not csv_files:
                raise FileNotFoundError("No CSV files found in MBS archive")
            
            logger.info(f"Found {len(csv_files)} MBS CSV files")
            
            # Process each CSV file and combine
            all_dataframes = []
            
            for csv_file in csv_files[:3]:  # Limit to first 3 files for performance
                logger.info(f"Processing MBS file: {csv_file.name}")
                
                try:
                    # Read CSV with Polars
                    df = pl.read_csv(
                        csv_file,
                        infer_schema_length=1000,
                        null_values=["", "NULL", "null", "N/A", "n/a"],
                        truncate_ragged_lines=True
                    )
                    
                    if len(df) > 0:
                        # Standardize column names (convert to lowercase with underscores)
                        standardized_columns = {}
                        for col in df.columns:
                            new_col = col.lower().replace(" ", "_").replace("-", "_")
                            standardized_columns[col] = new_col
                        
                        df = df.rename(standardized_columns)
                        
                        # Add source file info
                        df = df.with_columns(pl.lit(csv_file.name).alias("source_file"))
                        
                        all_dataframes.append(df)
                        logger.info(f"Loaded {len(df)} records from {csv_file.name}")
                    
                except Exception as e:
                    logger.warning(f"Failed to process {csv_file.name}: {e}")
                    continue
            
            if not all_dataframes:
                raise ValueError("No valid MBS data found")
            
            # Combine all dataframes
            combined_df = pl.concat(all_dataframes, how="diagonal")
            
            # Clean up temporary extraction
            import shutil
            shutil.rmtree(extract_dir)
            
            # Standardize and validate
            standardized_df = self._standardize_mbs_columns(combined_df)
            validated_df = self._validate_mbs_data(standardized_df)
            
            logger.info(f"✓ Successfully processed MBS data: {len(validated_df)} records")
            return validated_df
            
        except Exception as e:
            logger.error(f"Failed to extract MBS data: {e}")
            raise
    
    def extract_pbs_data(self, file_path: Path) -> pl.DataFrame:
        """
        Extract PBS (Pharmaceutical) data from CSV or ZIP file.
        
        Args:
            file_path: Path to PBS file
            
        Returns:
            Polars DataFrame with standardized PBS schema
        """
        if not self.validate_health_file(file_path, "pbs"):
            raise ValueError(f"Invalid PBS file: {file_path}")
        
        logger.info(f"Extracting PBS data from {file_path.name}")
        
        try:
            if file_path.suffix == ".csv":
                # Direct CSV processing
                df = pl.read_csv(
                    file_path,
                    infer_schema_length=1000,
                    null_values=["", "NULL", "null", "N/A", "n/a"],
                    truncate_ragged_lines=True
                )
                
            else:
                # ZIP file processing
                extract_dir = self.raw_dir / "temp_pbs"
                extract_dir.mkdir(exist_ok=True)
                
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                # Find main CSV file
                csv_files = list(extract_dir.rglob("*.csv"))
                if not csv_files:
                    raise FileNotFoundError("No CSV files found in PBS archive")
                
                # Use the largest CSV file (likely the main data)
                main_csv = max(csv_files, key=lambda f: f.stat().st_size)
                logger.info(f"Processing main PBS file: {main_csv.name}")
                
                df = pl.read_csv(
                    main_csv,
                    infer_schema_length=1000,
                    null_values=["", "NULL", "null", "N/A", "n/a"],
                    truncate_ragged_lines=True
                )
                
                # Clean up
                import shutil
                shutil.rmtree(extract_dir)
            
            # Standardize column names
            standardized_columns = {}
            for col in df.columns:
                new_col = col.lower().replace(" ", "_").replace("-", "_")
                standardized_columns[col] = new_col
            
            df = df.rename(standardized_columns)
            
            # Standardize and validate
            standardized_df = self._standardize_pbs_columns(df)
            validated_df = self._validate_pbs_data(standardized_df)
            
            logger.info(f"✓ Successfully processed PBS data: {len(validated_df)} records")
            return validated_df
            
        except Exception as e:
            logger.error(f"Failed to extract PBS data: {e}")
            raise
    
    def _standardize_mbs_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Standardize MBS column names and data types."""
        logger.info("Standardizing MBS column names and types")
        
        # Common MBS column mappings
        column_mappings = {
            # Temporal columns
            "year": "year",
            "calendar_year": "year",
            "quarter": "quarter",
            "month": "month",
            
            # Geographic columns  
            "state": "state",
            "postcode": "postcode",
            "state_code": "state",
            
            # Demographic columns
            "age_group": "age_group",
            "age": "age_group",
            "gender": "gender",
            "sex": "gender",
            
            # Service columns
            "item_number": "item_number",
            "item": "item_number",
            "category": "service_category",
            "service_category": "service_category",
            
            # Utilisation columns
            "services": "services_count",
            "services_count": "services_count",
            "number_of_services": "services_count",
            "benefit": "benefit_amount",
            "benefit_amount": "benefit_amount",
            "schedule_fee": "schedule_fee"
        }
        
        # Apply mappings where columns exist
        rename_dict = {}
        for old_col in df.columns:
            for pattern, new_col in column_mappings.items():
                if pattern in old_col.lower():
                    rename_dict[old_col] = new_col
                    break
        
        if rename_dict:
            df = df.rename(rename_dict)
            logger.info(f"Mapped {len(rename_dict)} MBS columns")
        
        # Select available columns that match our schema
        available_columns = [col for col in MBS_SCHEMA.keys() if col in df.columns]
        if available_columns:
            df = df.select(available_columns)
            logger.info(f"Selected {len(available_columns)} standardized MBS columns")
        
        return df
    
    def _standardize_pbs_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Standardize PBS column names and data types."""
        logger.info("Standardizing PBS column names and types")
        
        # Common PBS column mappings
        column_mappings = {
            "year": "year",
            "month": "month",
            "state": "state",
            "atc_code": "atc_code",
            "atc": "atc_code",
            "drug_name": "drug_name",
            "generic_name": "drug_name",
            "brand_name": "drug_name",
            "strength": "strength",
            "prescriptions": "prescriptions",
            "prescription_count": "prescriptions",
            "ddd_per_1000": "ddd_per_1000",
            "ddd_per_1000_pop_per_day": "ddd_per_1000",
            "cost_government": "cost_government",
            "government_cost": "cost_government",
            "cost_patient": "cost_patient",
            "patient_contribution": "cost_patient"
        }
        
        # Apply mappings
        rename_dict = {}
        for old_col in df.columns:
            for pattern, new_col in column_mappings.items():
                if pattern in old_col.lower():
                    rename_dict[old_col] = new_col
                    break
        
        if rename_dict:
            df = df.rename(rename_dict)
            logger.info(f"Mapped {len(rename_dict)} PBS columns")
        
        # Select available columns
        available_columns = [col for col in PBS_SCHEMA.keys() if col in df.columns]
        if available_columns:
            df = df.select(available_columns)
            logger.info(f"Selected {len(available_columns)} standardized PBS columns")
        
        return df
    
    def _validate_mbs_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Validate MBS data quality."""
        logger.info("Validating MBS data quality")
        
        initial_count = len(df)
        
        # Validate years (should be reasonable range)
        if "year" in df.columns:
            df = df.filter(
                (pl.col("year") >= 1990) & (pl.col("year") <= 2025)
            )
            logger.info(f"Filtered to valid years: {len(df)}")
        
        # Validate benefit amounts (should be positive)
        if "benefit_amount" in df.columns:
            df = df.filter(pl.col("benefit_amount") >= 0)
            logger.info(f"Filtered to valid benefit amounts: {len(df)}")
        
        final_count = len(df)
        logger.info(f"MBS validation complete: {initial_count} → {final_count} records")
        
        return df
    
    def _validate_pbs_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Validate PBS data quality."""
        logger.info("Validating PBS data quality")
        
        initial_count = len(df)
        
        # Validate years
        if "year" in df.columns:
            df = df.filter(
                (pl.col("year") >= 1990) & (pl.col("year") <= 2025)
            )
            logger.info(f"Filtered to valid years: {len(df)}")
        
        # Validate prescription counts (should be positive)
        if "prescriptions" in df.columns:
            df = df.filter(pl.col("prescriptions") > 0)
            logger.info(f"Filtered to valid prescription counts: {len(df)}")
        
        final_count = len(df)
        logger.info(f"PBS validation complete: {initial_count} → {final_count} records")
        
        return df
    
    def process_mbs_file(self, filename: Optional[str] = None) -> pl.DataFrame:
        """Process MBS file to analysis-ready DataFrame."""
        if filename is None:
            filename = MBS_CONFIG["historical_filename"]
        
        file_path = self.raw_dir / filename
        
        if not file_path.exists():
            logger.warning(f"MBS file not found: {file_path}, creating mock data")
            return self._create_mock_mbs_data()
        
        mbs_df = self.extract_mbs_data(file_path)
        
        # Export processed data
        output_path = self.processed_dir / "mbs_historical_processed.csv"
        mbs_df.write_csv(output_path)
        logger.info(f"Exported MBS data to {output_path}")
        
        return mbs_df
    
    def process_pbs_file(self, filename: Optional[str] = None) -> pl.DataFrame:
        """Process PBS file to analysis-ready DataFrame.""" 
        if filename is None:
            filename = PBS_CONFIG["current_filename"]
        
        file_path = self.raw_dir / filename
        
        if not file_path.exists():
            logger.warning(f"PBS file not found: {file_path}, creating mock data")
            return self._create_mock_pbs_data()
        
        pbs_df = self.extract_pbs_data(file_path)
        
        # Export processed data
        output_path = self.processed_dir / "pbs_current_processed.csv"
        pbs_df.write_csv(output_path)
        logger.info(f"Exported PBS data to {output_path}")
        
        return pbs_df
    
    def _create_mock_mbs_data(self) -> pl.DataFrame:
        """Create mock MBS data for testing."""
        logger.warning("Creating mock MBS data for testing")
        
        mock_data = {
            "year": [2015, 2015, 2015, 2014, 2014],
            "quarter": [1, 2, 3, 4, 4],
            "state": ["NSW", "VIC", "QLD", "SA", "WA"],
            "age_group": ["25-34", "35-44", "45-54", "55-64", "65-74"],
            "gender": ["M", "F", "M", "F", "M"],
            "item_number": ["23", "104", "36", "5020", "110"],
            "service_category": ["Professional Attendances", "Diagnostic Procedures", "Professional Attendances", "Operations", "Professional Attendances"],
            "services_count": [12500, 8900, 15600, 2300, 18900],
            "benefit_amount": [287500.50, 445600.25, 624800.75, 1250000.00, 378900.25]
        }
        
        return pl.DataFrame(mock_data)
    
    def _create_mock_pbs_data(self) -> pl.DataFrame:
        """Create mock PBS data for testing."""
        logger.warning("Creating mock PBS data for testing")
        
        mock_data = {
            "year": [2016, 2016, 2016, 2016, 2016],
            "month": [1, 2, 3, 4, 5],
            "state": ["NSW", "VIC", "QLD", "SA", "WA"],
            "atc_code": ["C09AA01", "A02BC01", "C10AA01", "N06AB03", "R03AC02"],
            "drug_name": ["Enalapril", "Omeprazole", "Simvastatin", "Fluoxetine", "Salbutamol"],
            "prescriptions": [125000, 89000, 156000, 67000, 234000],
            "ddd_per_1000": [45.2, 32.8, 67.5, 23.4, 89.1],
            "cost_government": [2875000.50, 1456000.25, 3248000.75, 1890000.00, 4567000.25],
            "cost_patient": [125000.00, 89000.50, 156000.25, 67000.75, 234000.30]
        }
        
        return pl.DataFrame(mock_data)
    
    def get_health_summary(self, mbs_df: pl.DataFrame, pbs_df: pl.DataFrame) -> Dict:
        """Generate summary statistics for health data."""
        summary = {
            "mbs_records": len(mbs_df),
            "pbs_records": len(pbs_df),
            "years_covered": [],
            "states_covered": [],
            "total_services": 0,
            "total_prescriptions": 0,
            "total_health_cost": 0
        }
        
        # Extract temporal coverage
        all_years = []
        if "year" in mbs_df.columns:
            all_years.extend(mbs_df["year"].unique().to_list())
        if "year" in pbs_df.columns:
            all_years.extend(pbs_df["year"].unique().to_list())
        
        summary["years_covered"] = sorted(set(all_years))
        
        # Extract geographic coverage
        all_states = []
        if "state" in mbs_df.columns:
            all_states.extend(mbs_df["state"].unique().to_list())
        if "state" in pbs_df.columns:
            all_states.extend(pbs_df["state"].unique().to_list())
        
        summary["states_covered"] = sorted(set(all_states))
        
        # Calculate totals
        if "services_count" in mbs_df.columns:
            summary["total_services"] = int(mbs_df["services_count"].sum())
        
        if "prescriptions" in pbs_df.columns:
            summary["total_prescriptions"] = int(pbs_df["prescriptions"].sum())
        
        # Calculate costs
        total_cost = 0
        if "benefit_amount" in mbs_df.columns:
            total_cost += float(mbs_df["benefit_amount"].sum())
        if "cost_government" in pbs_df.columns:
            total_cost += float(pbs_df["cost_government"].sum())
        
        summary["total_health_cost"] = total_cost
        
        return summary
    
    def process_complete_pipeline(self) -> Dict[str, pl.DataFrame]:
        """Execute complete health data processing pipeline."""
        console.print("🏥 [bold blue]Starting health data processing pipeline...[/bold blue]")
        
        try:
            # Process MBS data
            mbs_df = self.process_mbs_file()
            console.print(f"✅ Processed {len(mbs_df)} MBS records")
            
            # Process PBS data  
            pbs_df = self.process_pbs_file()
            console.print(f"✅ Processed {len(pbs_df)} PBS records")
            
            # Generate summary
            summary = self.get_health_summary(mbs_df, pbs_df)
            console.print(f"✅ Years covered: {summary['years_covered']}")
            console.print(f"✅ States: {summary['states_covered']}")
            console.print(f"✅ Total services: {summary['total_services']:,}")
            console.print(f"✅ Total prescriptions: {summary['total_prescriptions']:,}")
            console.print(f"✅ Total health cost: ${summary['total_health_cost']:,.2f}")
            
            console.print("🎉 [bold green]Health data processing pipeline complete![/bold green]")
            
            return {
                "mbs": mbs_df,
                "pbs": pbs_df,
                "summary": summary
            }
            
        except Exception as e:
            console.print(f"❌ [bold red]Health data processing failed: {e}[/bold red]")
            raise