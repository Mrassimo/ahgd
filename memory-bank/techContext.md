# Tech Context: AHGD ETL Refactoring

*This file lists technologies, development setup, constraints, and dependencies for the refactored AHGD ETL pipeline.*

## Core Technologies

*   **Programming Language:** Python (likely 3.9+ based on dependency versions)
*   **Key Libraries/Frameworks:**
    *   Polars: Core data manipulation and Parquet I/O.
    *   Pandas: Used alongside GeoPandas, potentially for specific tasks or legacy reasons.
    *   GeoPandas: Reading and processing ASGS shapefiles.
    *   Shapely: Underlying geometry operations for GeoPandas.
    *   Requests: Downloading data from URLs.
    *   PyArrow: Backend for Polars and Parquet support.
    *   Openpyxl: Reading Excel files (though not explicitly used in current geo/G01 logic, it's a dependency).
    *   Tqdm: Progress bars for downloads/extractions.
*   **Data Stores:**
    *   Input: ABS Website (via HTTPS URLs).
    *   Intermediate/Storage: Local Filesystem (ZIP files, extracted data).
    *   Output: Parquet files (now including `geo_dimension.parquet`, `population_dimension.parquet`, and `fact_assistance_need.parquet`).
*   **Orchestration:**
    *   Local: `run_etl.py` script with CLI arguments.
    *   Cloud: Google Colab notebooks (`colab_runner.ipynb`).
*   **Data Model:**
    *   Star Schema: Dimension tables (geo, future: demographic, time) and fact tables (assistance need, future: health condition, unpaid care).
    *   Surrogate Keys: Used in dimension tables and referenced as foreign keys in fact tables.

## Project Structure

The project has been migrated to a new clean structure with better separation of concerns:

### New Project Structure

- **Root Directory**: `/Users/massimoraso/AHGD/`
  - **`data_files/`**: All data input and output files
    - **`raw/`**: Downloaded raw data files
    - **`output/`**: Generated Parquet files 
  - **`app/`**: All code and documentation
    - **`etl_logic/`**: Core ETL processing modules
    - **`scripts/`**: Utility scripts
    - **`tests/`**: Test suite
    - **`documentation/`**: Project documentation
    - **`logs/`**: Log files
    - **`src/`**: Source modules
    - Root level files: `run_etl.py`, `.env`, `requirements.txt`, etc.

This structure provides a clean separation between data and code, making the project easier to maintain and understand.

## Development Setup

*   **Dependency Management:** `requirements.txt` (pip). `setup.py` defines the `etl_logic` package.
*   **Environment Management:** Python virtual environments recommended.
*   **Local Environment Configuration:** `.env` file via `python-dotenv` to manage `BASE_DIR`.
*   **Testing:** `pytest` framework in a `tests/` directory, with fixtures that simulate ABS file formats.
*   **Version Control:** Git (assumed, as project exists).

## Constraints

*   **Memory Usage:** GeoPandas can consume significant RAM when loading large shapefiles (e.g., SA1 level). Polars is generally more memory-efficient for tabular operations.
*   **Processing Time:** Downloading large ABS files and processing complex geometries can be time-consuming.
*   **Disk Space:** Raw ZIP files, extracted data, and output Parquet files require storage space (potentially several GBs).
*   **Local Setup:** Requires Python environment setup and dependency installation (`pip install -r requirements.txt`).
*   **Colab Environment:** Needs appropriate setup (installing dependencies, setting `BASE_DIR`).
*   **ABS Data Format Variations:** Column names in ABS Census files may vary, requiring flexible mapping logic.
*   **Path References:** After migration, code may need path updates to reflect the new structure.

## Key Dependencies (from `requirements.txt`)

*   `pandas>=2.0.0`
*   `geopandas>=0.13.0`
*   `polars>=0.20.0`
*   `requests>=2.31.0`
*   `tqdm>=4.66.0`
*   `pyarrow>=14.0.0`
*   `shapely>=2.0.0`
*   `openpyxl>=3.1.0`
*   `pytest>=7.0.0` (added for testing)
*   `pytest-mock>=3.10.0` (added for testing)
*   `python-dotenv>=1.0.0` (added for environment configuration)

## Data Sources

*   **ABS ASGS 2021 Shapefiles:** SA1, SA2, SA3, SA4, STATE boundaries (GDA2020). URLs defined in `direct_urls.py`.
*   **ABS Census 2021 Data Packs:** 
    *   G01 (Selected Person Characteristics) from the "all for AUS short-header" pack.
    *   G17 (Core Activity Need for Assistance) from the same pack.
    *   URLs defined in `direct_urls.py`.

## Output Structure

The ETL pipeline now produces the following output files:

*   **geo_dimension.parquet:**
    *   Dimension table with geographic boundaries
    *   Contains geo_sk (surrogate key), geo_code, geo_level, geometry_wkt

*   **population_dimension.parquet:**
    *   Legacy structure (not yet fully aligned with star schema)
    *   Contains geo_code, total_persons, total_male, total_female, total_indigenous
    *   Will eventually be refactored to use surrogate keys and split into proper dimension/fact tables

*   **fact_assistance_need.parquet:**
    *   Fact table implementing star schema pattern
    *   Contains geo_sk (foreign key to geo_dimension), assistance_needed_count, no_assistance_needed_count, assistance_not_stated_count
    *   Represents census G17 data linked to geography via surrogate key

## Metadata Files

The project relies on metadata files from ABS to understand the structure of Census data:

### Key Metadata Files

1. **Metadata_2021_GCP_DataPack_R1_R2.xlsx**
   - Location: `data/raw/Metadata/`
   - Purpose: Provides detailed descriptions of Census tables
   - Usage: Reference for table purposes and population segments

2. **2021_GCP_Sequential_Template_R2.xlsx**
   - Location: `data/raw/Metadata/`
   - Purpose: Shows exact structure and layout of Census tables
   - Usage: Reference for column patterns and data organization

3. **2021Census_geog_desc_1st_2nd_3rd_release.xlsx**
   - Location: `data/raw/Metadata/`
   - Purpose: Documents geographic structures and codes
   - Usage: Reference for geographic hierarchy and code formats

### How Metadata Files Are Used

1. **Development Process**:
   - Analyzed metadata files to understand table structures
   - Identified column naming patterns and variations
   - Documented these patterns in `dataStructures.md`
   - Implemented flexible column mapping based on identified patterns

2. **Runtime Process**:
   - Metadata files are not used at runtime
   - Instead, the patterns and knowledge derived from them are embedded in the code
   - This allows the ETL process to handle variations without direct metadata reference

3. **Documentation**:
   - Added detailed README in `data/raw/Metadata/`
   - Created `dataStructures.md` in the Memory Bank
   - Updated system patterns to document metadata-driven approach

## File Structure

### Key Directories

- **src/etl/**: Core ETL processing modules
- **tests/**: Test suite
- **data/**: Raw and processed data
  - **data/raw/**: Raw Census and geographic data
  - **data/raw/Metadata/**: Census metadata files 
  - **data/processed/**: Output of ETL processes
- **memory-bank/**: Project documentation

### Key Files

- **run_etl.py**: Main ETL orchestrator
- **src/etl/config.py**: Configuration and constants
- **src/etl/census.py**: Census data processing
- **src/etl/geography.py**: Geographic data processing
- **src/etl/utils.py**: Utility functions
- **src/etl/process_g17.py**: G17 (Personal Income) specific processing
- **src/etl/process_g18.py**: G18 (Core Activity Need for Assistance) specific processing
- **src/etl/process_g19.py**: G19 (Long-Term Health Conditions) specific processing

## Data Processing Strategy

### Metadata-Informed Development

Our development approach uses metadata analysis to guide implementation:

1. **Metadata Analysis Phase**:
   - Examine Excel metadata files to understand structure
   - Develop scripts (`extract_metadata.py`, `extract_g18_g19_info.py`) to explore table structures
   - Document findings and patterns

2. **ETL Implementation Phase**:
   - Apply knowledge from metadata to develop robust ETL processes
   - Implement flexible column matching to handle inconsistencies
   - Standardize output formats based on identified patterns

3. **Testing Phase**:
   - Create tests that validate handling of column variations
   - Ensure ETL processes correctly handle metadata-documented edge cases
   - Verify output standardization meets requirements

This metadata-informed approach enables us to handle the complexities of ABS Census data without requiring the metadata files at runtime.

## Constraints and Considerations

- **Data Volume**: Census files can be large, requiring efficient processing
- **Column Name Inconsistencies**: ABS data has typos and variations that must be handled
- **Split Files**: Some tables (e.g., G19) are split across multiple files due to column limits
- **Environment Compatibility**: Code must work across local and Colab environments