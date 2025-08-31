"""
⚠️  DEPRECATED: Legacy DLT Pipeline for Health Service Data

⚠️  This pandas-based pipeline has been REPLACED by health_polars.py
⚠️  New pipeline provides 10-100x performance improvement with Polars
⚠️  This file will be removed in a future version

For new implementations, use:
    from pipelines.dlt.health_polars import load_health_data_polars

Legacy functionality (DEPRECATED):
- Medicare Benefits Schedule (MBS) data
- Pharmaceutical Benefits Scheme (PBS) data  
- AIHW mortality data (MORT/GRIM)
- PHIDU chronic disease prevalence data
"""

import logging
import requests
import pandas as pd
import zipfile
import io
from typing import Iterator, Dict, Any, Optional, List
from pathlib import Path
import dlt
from datetime import datetime

from src.models.health import MBSRecord, PBSRecord, AIHWMortalityRecord, PHIDUChronicDiseaseRecord
from src.utils.geographic import GeographicMatcher

logger = logging.getLogger(__name__)

# Data sources from REAL_DATA_SOURCES.md
MBS_HISTORICAL_URL = "https://data.gov.au/data/dataset/8a19a28f-35b0-4035-8cd5-5b611b3cfa6f/resource/519b55ab-8f81-47d1-a483-8495668e38d8/download/mbs-demographics-historical-1993-2015.zip"
PBS_CURRENT_URL = "https://data.gov.au/data/dataset/14b536d4-eb6a-485d-bf87-2e6e77ddbac1/resource/08eda5ab-01c0-4c94-8b1a-157bcffe80d3/download/pbs-item-2016csvjuly.csv"
PBS_HISTORICAL_URL = "https://data.gov.au/data/dataset/14b536d4-eb6a-485d-bf87-2e6e77ddbac1/resource/56f87bbb-a7cb-4cbf-a723-7aec22996eee/download/csv-pbs-item-historical-1992-2014.zip"

AIHW_MORT_TABLE1_URL = "https://data.gov.au/data/dataset/a84a6e8e-dd8f-4bae-a79d-77a5e32877ad/resource/a5de4e7e-d062-4356-9d1b-39f44b1961dc/download/aihw-phe-229-mort-table1-data-gov-au-2025.csv"
AIHW_GRIM_URL = "https://data.gov.au/data/dataset/488ef6d4-c763-4b24-b8fb-9c15b67ece19/resource/edcbc14c-ba7c-44ae-9d4f-2622ad3fafe0/download/aihw-phe-229-grim-data-gov-au-2025.csv"

PHIDU_PHA_URL = "https://phidu.torrens.edu.au/current/data/sha-aust/pha/phidu_data_pha_aust.xlsx"


@dlt.source(name="health_data")
def health_data_source():
    """DLT source for Australian health service data."""
    return [
        mbs_data_resource(),
        pbs_data_resource(),
        aihw_mortality_resource(),
        phidu_chronic_disease_resource()
    ]


def download_and_extract_zip(url: str, target_dir: Path = None) -> List[Path]:
    """Download and extract ZIP files, return list of extracted file paths."""
    logger.info(f"Downloading ZIP from {url}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    if target_dir is None:
        target_dir = Path("data/temp")
        target_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_files = []
    
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        for file_info in zip_ref.filelist:
            if file_info.filename.endswith('.csv'):
                extracted_path = target_dir / file_info.filename
                with open(extracted_path, 'wb') as f:
                    f.write(zip_ref.read(file_info.filename))
                extracted_files.append(extracted_path)
                logger.info(f"Extracted: {extracted_path}")
    
    return extracted_files


def download_csv(url: str, target_path: Path = None) -> Path:
    """Download CSV file directly."""
    logger.info(f"Downloading CSV from {url}")
    
    if target_path is None:
        target_path = Path("data/temp") / url.split('/')[-1]
        target_path.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(target_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    logger.info(f"Downloaded: {target_path}")
    return target_path


@dlt.resource(name="mbs_data", write_disposition="merge", primary_key=["mbs_item_number", "geographic_code", "age_group", "gender", "financial_year"])
def mbs_data_resource() -> Iterator[Dict[str, Any]]:
    """
    Extract and validate MBS health service utilisation data.
    
    Downloads MBS demographics data and processes it for SA1-level analysis
    through geographic aggregation and population weighting.
    """
    logger.info("Starting MBS data extraction")
    
    try:
        # Download MBS historical data (ZIP file)
        zip_files = download_and_extract_zip(MBS_HISTORICAL_URL)
        
        geo_matcher = GeographicMatcher()
        processed_count = 0
        
        for file_path in zip_files:
            logger.info(f"Processing MBS file: {file_path}")
            
            # Read CSV with appropriate encoding
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin1')
            
            # Process in chunks to manage memory
            chunk_size = 5000
            for chunk_start in range(0, len(df), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(df))
                chunk = df.iloc[chunk_start:chunk_end]
                
                for _, row in chunk.iterrows():
                    try:
                        # Map geographic areas to SA1 level
                        sa1_mappings = geo_matcher.map_to_sa1(
                            row.get('postcode') or row.get('lga_code') or row.get('sa3_code'),
                            source_type='auto'
                        )
                        
                        for sa1_code, weight in sa1_mappings:
                            # Create MBS record with Pydantic validation
                            record = MBSRecord(
                                geographic_code=sa1_code,
                                geographic_name=geo_matcher.get_sa1_name(sa1_code),
                                state_code=str(sa1_code)[0],  # First digit is state
                                mbs_item_number=str(row.get('item_number', '')),
                                mbs_item_description=str(row.get('item_description', 'Unknown')),
                                service_type=_classify_service_type(row.get('item_description', '')),
                                age_group=_map_age_group(row.get('age_group', 'ALL')),
                                gender=_map_gender(row.get('gender', 'ALL')),
                                service_count=int(row.get('service_count', 0) * weight),
                                patient_count=int(row.get('patient_count', 0) * weight) if row.get('patient_count') else None,
                                benefit_paid=float(row.get('benefit_paid', 0.0) * weight),
                                financial_year=row.get('financial_year', '2015-16'),
                                quarter=row.get('quarter') if row.get('quarter') != 'ALL' else None,
                                quality_score=0.95,  # High quality for government data
                                source_system='MBS_HISTORICAL',
                                last_updated=datetime.now()
                            )
                            
                            yield record.model_dump()
                            processed_count += 1
                            
                            if processed_count % 1000 == 0:
                                logger.info(f"Processed {processed_count} MBS records")
                                
                    except Exception as e:
                        logger.warning(f"Failed to process MBS row: {e}")
                        continue
        
        logger.info(f"MBS data extraction completed. Total records: {processed_count}")
        
    except Exception as e:
        logger.error(f"MBS data extraction failed: {e}")
        raise


@dlt.resource(name="pbs_data", write_disposition="merge", primary_key=["pbs_item_code", "geographic_code", "age_group", "gender", "financial_year"])
def pbs_data_resource() -> Iterator[Dict[str, Any]]:
    """
    Extract and validate PBS pharmaceutical utilisation data.
    
    Downloads both current and historical PBS data for comprehensive
    pharmaceutical usage analysis at SA1 level.
    """
    logger.info("Starting PBS data extraction")
    
    try:
        geo_matcher = GeographicMatcher()
        processed_count = 0
        
        # Process current PBS data
        current_file = download_csv(PBS_CURRENT_URL)
        df_current = pd.read_csv(current_file)
        
        processed_count += yield from _process_pbs_dataframe(
            df_current, geo_matcher, "PBS_CURRENT"
        )
        
        # Process historical PBS data
        historical_files = download_and_extract_zip(PBS_HISTORICAL_URL)
        
        for file_path in historical_files:
            logger.info(f"Processing PBS historical file: {file_path}")
            
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin1')
            
            processed_count += yield from _process_pbs_dataframe(
                df, geo_matcher, "PBS_HISTORICAL"
            )
        
        logger.info(f"PBS data extraction completed. Total records: {processed_count}")
        
    except Exception as e:
        logger.error(f"PBS data extraction failed: {e}")
        raise


def _process_pbs_dataframe(df: pd.DataFrame, geo_matcher: GeographicMatcher, source: str) -> Iterator[Dict[str, Any]]:
    """Process PBS DataFrame and yield validated records."""
    count = 0
    
    chunk_size = 5000
    for chunk_start in range(0, len(df), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(df))
        chunk = df.iloc[chunk_start:chunk_end]
        
        for _, row in chunk.iterrows():
            try:
                # Map to SA1 level
                sa1_mappings = geo_matcher.map_to_sa1(
                    row.get('postcode') or row.get('lga_code'),
                    source_type='auto'
                )
                
                for sa1_code, weight in sa1_mappings:
                    record = PBSRecord(
                        geographic_code=sa1_code,
                        geographic_name=geo_matcher.get_sa1_name(sa1_code),
                        state_code=str(sa1_code)[0],
                        pbs_item_code=str(row.get('item_code', '')),
                        medicine_name=str(row.get('medicine_name', 'Unknown')),
                        brand_name=row.get('brand_name'),
                        atc_code=row.get('atc_code'),
                        therapeutic_group=row.get('therapeutic_group'),
                        age_group=_map_age_group(row.get('age_group', 'ALL')),
                        gender=_map_gender(row.get('gender', 'ALL')),
                        prescription_count=int(row.get('prescription_count', 0) * weight),
                        patient_count=int(row.get('patient_count', 0) * weight) if row.get('patient_count') else None,
                        government_benefit=float(row.get('government_benefit', 0.0) * weight),
                        patient_contribution=float(row.get('patient_contribution', 0.0) * weight) if row.get('patient_contribution') else None,
                        financial_year=row.get('financial_year', '2016-17'),
                        month=row.get('month') if row.get('month') != 'ALL' else None,
                        quality_score=0.95,
                        source_system=source,
                        last_updated=datetime.now()
                    )
                    
                    yield record.model_dump()
                    count += 1
                    
                    if count % 1000 == 0:
                        logger.info(f"Processed {count} PBS records")
                        
            except Exception as e:
                logger.warning(f"Failed to process PBS row: {e}")
                continue
    
    return count


@dlt.resource(name="aihw_mortality", write_disposition="merge", primary_key=["geographic_code", "cause_of_death", "age_group", "gender", "calendar_year"])
def aihw_mortality_resource() -> Iterator[Dict[str, Any]]:
    """
    Extract and validate AIHW mortality data from MORT and GRIM datasets.
    
    Processes death counts, rates, and mortality indicators with
    comprehensive cause-of-death classification.
    """
    logger.info("Starting AIHW mortality data extraction")
    
    try:
        geo_matcher = GeographicMatcher()
        processed_count = 0
        
        # Process MORT Table 1 data
        mort_file = download_csv(AIHW_MORT_TABLE1_URL)
        df_mort = pd.read_csv(mort_file)
        
        processed_count += yield from _process_mort_dataframe(
            df_mort, geo_matcher, "MORT"
        )
        
        # Process GRIM data
        grim_file = download_csv(AIHW_GRIM_URL)
        df_grim = pd.read_csv(grim_file)
        
        processed_count += yield from _process_grim_dataframe(
            df_grim, geo_matcher, "GRIM"
        )
        
        logger.info(f"AIHW mortality data extraction completed. Total records: {processed_count}")
        
    except Exception as e:
        logger.error(f"AIHW mortality data extraction failed: {e}")
        raise


def _process_mort_dataframe(df: pd.DataFrame, geo_matcher: GeographicMatcher, source: str) -> Iterator[Dict[str, Any]]:
    """Process MORT DataFrame and yield validated records."""
    count = 0
    
    for _, row in df.iterrows():
        try:
            # Map SA3/SA4/LGA to SA1 level
            sa1_mappings = geo_matcher.map_to_sa1(
                row.get('geographic_code'),
                source_type=row.get('geographic_level', 'SA3')
            )
            
            for sa1_code, weight in sa1_mappings:
                record = AIHWMortalityRecord(
                    geographic_code=sa1_code,
                    geographic_name=geo_matcher.get_sa1_name(sa1_code),
                    state_code=str(sa1_code)[0],
                    cause_of_death=_map_cause_of_death(row.get('cause_category', 'ALL_CAUSES')),
                    icd_10_code=row.get('icd_10_code'),
                    cause_description=row.get('cause_description'),
                    age_group=_map_age_group(row.get('age_group', 'ALL')),
                    gender=_map_gender(row.get('gender', 'ALL')),
                    death_count=int(row.get('death_count', 0) * weight),
                    crude_death_rate=float(row.get('crude_rate', 0.0)) if row.get('crude_rate') else None,
                    age_standardised_rate=float(row.get('age_std_rate', 0.0)) if row.get('age_std_rate') else None,
                    calendar_year=int(row.get('year', 2023)),
                    data_source=source,
                    quality_score=0.98,  # Very high quality for AIHW data
                    source_system='AIHW_MORT',
                    last_updated=datetime.now()
                )
                
                yield record.model_dump()
                count += 1
                
                if count % 1000 == 0:
                    logger.info(f"Processed {count} MORT records")
                    
        except Exception as e:
            logger.warning(f"Failed to process MORT row: {e}")
            continue
    
    return count


def _process_grim_dataframe(df: pd.DataFrame, geo_matcher: GeographicMatcher, source: str) -> Iterator[Dict[str, Any]]:
    """Process GRIM DataFrame and yield validated records."""
    count = 0
    
    for _, row in df.iterrows():
        try:
            # GRIM data is typically national level, distribute across all SA1s
            # or use available geographic indicators
            geographic_code = row.get('geographic_code') or 'NATIONAL'
            
            if geographic_code == 'NATIONAL':
                # For national data, we might skip or handle differently
                # For now, we'll create a single national-level record
                national_sa1_code = '10000000000'  # Placeholder national SA1
                sa1_mappings = [(national_sa1_code, 1.0)]
            else:
                sa1_mappings = geo_matcher.map_to_sa1(geographic_code, source_type='auto')
            
            for sa1_code, weight in sa1_mappings:
                record = AIHWMortalityRecord(
                    geographic_code=sa1_code,
                    geographic_name=geo_matcher.get_sa1_name(sa1_code),
                    state_code=str(sa1_code)[0] if len(sa1_code) >= 11 else '0',
                    cause_of_death=_map_cause_of_death(row.get('cause_category', 'ALL_CAUSES')),
                    icd_10_code=row.get('icd_10_code'),
                    cause_description=row.get('cause_description'),
                    age_group=_map_age_group(row.get('age_group', 'ALL')),
                    gender=_map_gender(row.get('gender', 'ALL')),
                    death_count=int(row.get('death_count', 0) * weight),
                    crude_death_rate=float(row.get('crude_rate', 0.0)) if row.get('crude_rate') else None,
                    age_standardised_rate=float(row.get('age_std_rate', 0.0)) if row.get('age_std_rate') else None,
                    calendar_year=int(row.get('year', 2023)),
                    data_source=source,
                    quality_score=0.95,  # High quality for AIHW GRIM data
                    source_system='AIHW_GRIM',
                    last_updated=datetime.now()
                )
                
                yield record.model_dump()
                count += 1
                
                if count % 1000 == 0:
                    logger.info(f"Processed {count} GRIM records")
                    
        except Exception as e:
            logger.warning(f"Failed to process GRIM row: {e}")
            continue
    
    return count


@dlt.resource(name="phidu_chronic_disease", write_disposition="merge", primary_key=["geographic_code", "disease_type", "age_group", "gender"])
def phidu_chronic_disease_resource() -> Iterator[Dict[str, Any]]:
    """
    Extract and validate PHIDU chronic disease prevalence data.
    
    Downloads PHIDU Social Health Atlas data and processes the complex
    multi-sheet Excel structure for SA1-level analysis.
    """
    logger.info("Starting PHIDU chronic disease data extraction")
    
    try:
        # Download PHIDU data (large Excel file)
        target_path = Path("data/temp/phidu_data_pha_aust.xlsx")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading PHIDU data (73.7 MB) from {PHIDU_PHA_URL}")
        response = requests.get(PHIDU_PHA_URL, stream=True)
        response.raise_for_status()
        
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        geo_matcher = GeographicMatcher()
        processed_count = 0
        
        # Process multiple sheets in PHIDU Excel file
        excel_file = pd.ExcelFile(target_path)
        
        for sheet_name in excel_file.sheet_names:
            if any(keyword in sheet_name.lower() for keyword in ['chronic', 'disease', 'prevalence']):
                logger.info(f"Processing PHIDU sheet: {sheet_name}")
                
                df = pd.read_excel(target_path, sheet_name=sheet_name)
                processed_count += yield from _process_phidu_dataframe(
                    df, geo_matcher, sheet_name
                )
        
        logger.info(f"PHIDU data extraction completed. Total records: {processed_count}")
        
    except Exception as e:
        logger.error(f"PHIDU data extraction failed: {e}")
        raise


def _process_phidu_dataframe(df: pd.DataFrame, geo_matcher: GeographicMatcher, sheet_name: str) -> Iterator[Dict[str, Any]]:
    """Process PHIDU DataFrame and yield validated records."""
    count = 0
    
    for _, row in df.iterrows():
        try:
            # Map PHA to SA1 level using population weights
            pha_code = row.get('pha_code')
            sa1_mappings = geo_matcher.map_pha_to_sa1(pha_code)
            
            for sa1_code, weight in sa1_mappings:
                record = PHIDUChronicDiseaseRecord(
                    geographic_code=sa1_code,
                    geographic_name=geo_matcher.get_sa1_name(sa1_code),
                    state_code=str(sa1_code)[0],
                    disease_type=_extract_disease_type(sheet_name),
                    disease_description=sheet_name,
                    prevalence_rate=float(row.get('prevalence_rate', 0.0)),
                    age_group=_map_age_group(row.get('age_group', 'ALL')),
                    gender=_map_gender(row.get('gender', 'ALL')),
                    pha_code=pha_code,
                    pha_name=row.get('pha_name'),
                    sa2_mapping_percentage=weight * 100,
                    population_total=int(row.get('population', 0)),
                    quality_score=0.90,  # High quality but some mapping uncertainty
                    source_system='PHIDU',
                    last_updated=datetime.now()
                )
                
                yield record.model_dump()
                count += 1
                
                if count % 500 == 0:
                    logger.info(f"Processed {count} PHIDU records from {sheet_name}")
                    
        except Exception as e:
            logger.warning(f"Failed to process PHIDU row: {e}")
            continue
    
    return count


# Helper methods for data mapping
def _classify_service_type(description: str) -> str:
    """Classify MBS service type from description."""
    description = description.upper()
    
    if any(term in description for term in ['CONSULT', 'VISIT', 'EXAMINATION']):
        return 'MEDICAL'
    elif any(term in description for term in ['X-RAY', 'SCAN', 'ULTRASOUND', 'MRI']):
        return 'DIAGNOSTIC'
    elif any(term in description for term in ['PATHOLOGY', 'BLOOD', 'URINE', 'TEST']):
        return 'PATHOLOGY'
    elif any(term in description for term in ['SURGERY', 'OPERATION', 'PROCEDURE']):
        return 'SURGICAL'
    elif any(term in description for term in ['MENTAL', 'PSYCHIATR', 'PSYCHOLOGY']):
        return 'MENTAL_HEALTH'
    else:
        return 'MEDICAL'


def _map_age_group(age_group: str) -> str:
    """Map various age group formats to standard categories."""
    if not age_group or age_group.upper() == 'ALL':
        return 'ALL_AGES'
    
    age_mappings = {
        '0-1': 'INFANT',
        '2-12': 'CHILD',
        '13-17': 'ADOLESCENT',
        '18-24': 'YOUNG_ADULT',
        '25-44': 'ADULT',
        '45-64': 'MIDDLE_AGE',
        '65-74': 'OLDER_ADULT',
        '75+': 'ELDERLY'
    }
    
    return age_mappings.get(age_group, 'ALL_AGES')


def _map_gender(gender: str) -> str:
    """Map various gender formats to standard categories."""
    if not gender or gender.upper() in ['ALL', 'TOTAL']:
        return 'ALL'
    
    gender = gender.upper()
    if gender in ['M', 'MALE', 'MALES']:
        return 'MALE'
    elif gender in ['F', 'FEMALE', 'FEMALES']:
        return 'FEMALE'
    else:
        return 'ALL'


def _map_cause_of_death(cause: str) -> str:
    """Map cause of death to standard categories."""
    if not cause:
        return 'ALL_CAUSES'
    
    cause = cause.upper()
    cause_mappings = {
        'CANCER': 'CANCER',
        'CARDIOVASCULAR': 'CARDIOVASCULAR',
        'RESPIRATORY': 'RESPIRATORY',
        'DIABETES': 'DIABETES',
        'MENTAL': 'MENTAL_HEALTH',
        'SUICIDE': 'SUICIDE',
        'ACCIDENT': 'ACCIDENT',
        'DEMENTIA': 'DEMENTIA'
    }
    
    for key, value in cause_mappings.items():
        if key in cause:
            return value
    
    return 'OTHER'


def _extract_disease_type(sheet_name: str) -> str:
    """Extract disease type from PHIDU sheet name."""
    sheet_name = sheet_name.upper()
    
    disease_mappings = {
        'DIABETES': 'DIABETES',
        'CARDIOVASCULAR': 'CARDIOVASCULAR',
        'CANCER': 'CANCER',
        'MENTAL': 'MENTAL_HEALTH',
        'RESPIRATORY': 'RESPIRATORY',
        'ARTHRITIS': 'ARTHRITIS',
        'KIDNEY': 'KIDNEY_DISEASE',
        'DEMENTIA': 'DEMENTIA'
    }
    
    for key, value in disease_mappings.items():
        if key in sheet_name:
            return value
    
    return 'CARDIOVASCULAR'  # Default for unknown


# Main pipeline functions
def load_mbs_pbs_data():
    """
    ⚠️  DEPRECATED: Load MBS/PBS health service utilisation data.
    
    This function is deprecated and will be removed. Use:
        from pipelines.dlt.health_polars import load_health_data_polars
    
    New pipeline provides 10-100x performance improvement.
    """
    import warnings
    warnings.warn(
        "load_mbs_pbs_data() is deprecated. Use load_health_data_polars() for 10-100x performance improvement.",
        DeprecationWarning,
        stacklevel=2
    )
    logger.warning("⚠️  Using deprecated pandas pipeline. Switch to health_polars.py for massive performance gains!")
    logger.info("Starting legacy MBS/PBS data pipeline")
    
    pipeline = dlt.pipeline(
        pipeline_name="mbs_pbs_health_data",
        destination="duckdb",
        dataset_name="health_analytics"
    )
    
    # Load MBS and PBS data
    load_info = pipeline.run([mbs_data_resource(), pbs_data_resource()])
    logger.info(f"MBS/PBS pipeline completed: {load_info}")
    
    return {"status": "completed", "load_info": str(load_info)}


def load_aihw_mortality_data():
    """
    ⚠️  DEPRECATED: Load AIHW mortality data from MORT/GRIM datasets.
    
    This function is deprecated and will be removed. Use:
        from pipelines.dlt.health_polars import load_health_data_polars
    """
    import warnings
    warnings.warn(
        "load_aihw_mortality_data() is deprecated. Use load_health_data_polars() for 10-100x performance improvement.",
        DeprecationWarning,
        stacklevel=2
    )
    logger.warning("⚠️  Using deprecated pandas pipeline. Switch to health_polars.py for massive performance gains!")
    logger.info("Starting legacy AIHW mortality data pipeline")
    
    pipeline = dlt.pipeline(
        pipeline_name="aihw_mortality_data",
        destination="duckdb",
        dataset_name="health_analytics"
    )
    
    load_info = pipeline.run([aihw_mortality_resource()])
    logger.info(f"AIHW mortality pipeline completed: {load_info}")
    
    return {"status": "completed", "load_info": str(load_info)}


def load_phidu_chronic_disease_data():
    """
    ⚠️  DEPRECATED: Load PHIDU chronic disease prevalence data.
    
    This function is deprecated and will be removed. Use:
        from pipelines.dlt.health_polars import load_health_data_polars
    """
    import warnings
    warnings.warn(
        "load_phidu_chronic_disease_data() is deprecated. Use load_health_data_polars() for 10-100x performance improvement.",
        DeprecationWarning,
        stacklevel=2
    )
    logger.warning("⚠️  Using deprecated pandas pipeline. Switch to health_polars.py for massive performance gains!")
    logger.info("Starting legacy PHIDU chronic disease data pipeline")
    
    pipeline = dlt.pipeline(
        pipeline_name="phidu_chronic_disease_data",
        destination="duckdb",
        dataset_name="health_analytics"
    )
    
    load_info = pipeline.run([phidu_chronic_disease_resource()])
    logger.info(f"PHIDU chronic disease pipeline completed: {load_info}")
    
    return {"status": "completed", "load_info": str(load_info)}