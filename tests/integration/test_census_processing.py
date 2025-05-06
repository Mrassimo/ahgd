import pytest
import logging
from pandas.testing import assert_frame_equal
import polars as pl
from polars.testing import assert_frame_equal
from pathlib import Path
import zipfile
import logging

# Corrected imports relative to project structure
from etl_logic import utils
from etl_logic.tables import g01_population, g19_health_conditions

@pytest.fixture(scope="function")
def sample_health_dim(temp_etl_dirs):
    """Create and save a sample health condition dimension."""
    # Based on columns parsed in test_g19_health_conditions.py
    health_df = pl.DataFrame({
        "condition_sk": [10, 20, 30, 40],
        "condition_code": ["arthritis", "asthma", "not_stated", "no_condition"],
        "condition_label": ["Arthritis", "Asthma", "Not Stated", "No Condition"]
    })
    health_path = temp_etl_dirs["dims"] / "dim_health_condition.parquet"
    health_df.write_parquet(health_path)
    return health_path

@pytest.fixture(scope="function")
def sample_demo_dim(temp_etl_dirs):
    """Create and save a sample demographic dimension.
    Needed for G19 as it processes Sex and Age Range.
    """
    # Based on columns parsed in test_g19_health_conditions.py
    # We need combinations appearing in the unpivoted G19 data
    demo_df = pl.DataFrame({
        "demographic_sk": [101, 102, 103, 104, 105],
        "sex": ["P", "F", "M", "P", "P"], # Corresponds to P_Arthritis, F_Asthma, M_NS, P_No_condition
        "age_group": ["0-14", "15-24", "85+", "total", "total"] # Simplified age group mapping
    })
    demo_path = temp_etl_dirs["dims"] / "dim_demographic.parquet"
    demo_df.write_parquet(demo_path)
    return demo_path

@pytest.fixture(scope="function")
def create_dummy_zip():
    """Factory fixture to create a dummy zip file with specified content."""
    # ... existing code ...

def test_process_census_table_g19(temp_etl_dirs, sample_geo_dim, sample_time_dim,
                                sample_health_dim, sample_demo_dim, # Add new dimension fixtures
                                create_dummy_zip, caplog):
    """Integration test for processing G19 using utils.process_census_table."""
    # Arrange
    table_code = "G19"
    csv_name = f"{table_code}_SA1_TEST.csv"
    zip_name = f"2021_GCP_{table_code}_SA1_for_AUS.zip"
    zip_path = temp_etl_dirs["zip"] / zip_name
    output_filename = f"fact_health_conditions_refined.parquet" # Final expected name for G19
    output_path = temp_etl_dirs["output"] / output_filename

    create_dummy_zip(zip_path, csv_name, G19_CSV_CONTENT)

    # Define dimension paths required by process_census_table for joins
    # Check the signature/logic of process_census_table to confirm these names
    dimension_paths = {
        "geo": sample_geo_dim,
        "time": sample_time_dim, # Passed as SK, but let's assume path might be needed elsewhere
        "health_condition": sample_health_dim,
        "demographic": sample_demo_dim
        # Add characteristic dim if needed
    }

    # Act
    with caplog.at_level(logging.INFO):
        # Assuming process_census_table handles finding/joining dimensions based on convention or args
        # We might need to adjust the call signature or mock dimension loading if it's complex
        success = utils.process_census_table(
            table_code=table_code,
            process_file_function=g19_health_conditions.process_g19_file, # Use the G19 specific processor
            output_filename=output_filename,
            zip_dir=temp_etl_dirs["zip"],
            temp_extract_base=temp_etl_dirs["extract"],
            output_dir=temp_etl_dirs["output"],
            geo_output_path=sample_geo_dim,
            time_sk=sample_time_dim,
            # Pass paths to other required dimensions if the function expects them
            # These might be needed for the join step *after* process_g19_file runs
            dim_paths=dimension_paths # Assuming a dict arg like this exists, adjust if needed
        )

    # Assert
    assert success is True
    assert output_path.exists(), f"Output file {output_path} was not created."
    # Check logs specific to G19 processing and joins
    assert f"Found 1 zip files for table code {table_code}" in caplog.text
    assert f"Successfully processed {csv_name}" in caplog.text
    assert f"Joined with geo dimension" in caplog.text
    assert f"Added time_sk {sample_time_dim}" in caplog.text
    # Check logs for other dimension joins if process_census_table logs them
    assert f"Joining intermediate data with health condition dimension" in caplog.text
    assert f"Joining intermediate data with demographic dimension" in caplog.text
    assert f"Successfully wrote final fact table to {output_path}" in caplog.text

    # Verify output content
    result_df = pl.read_parquet(output_path)

    # Expected output based on G19_CSV_CONTENT, unpivoted by process_g19_file,
    # and joined with sample dimensions.
    # geo_sk: 1 for 101011001, 2 for 101011002
    # time_sk: 202101
    # condition_sk: 10 (arthritis), 20 (asthma), 30 (not_stated), 40 (no_condition)
    # demographic_sk: 101 (P, 0-14), 102 (F, 15-24), 103 (M, 85+), 104 (P, total), 105 (P, total) - SKs need to match the join keys
    expected_data = [
        # Geo 101011001 (geo_sk=1)
        {"geo_sk": 1, "time_sk": sample_time_dim, "condition_sk": 10, "demographic_sk": 101, "count": 2}, # P_Arthritis_0_14
        {"geo_sk": 1, "time_sk": sample_time_dim, "condition_sk": 20, "demographic_sk": 102, "count": 5}, # F_Asthma_15_24
        {"geo_sk": 1, "time_sk": sample_time_dim, "condition_sk": 30, "demographic_sk": 103, "count": 1}, # M_NS_85_over
        {"geo_sk": 1, "time_sk": sample_time_dim, "condition_sk": 40, "demographic_sk": 104, "count": 42}, # P_No_condition_Tot (mapped to demo_sk 104 or 105 based on join)
        # Geo 101011002 (geo_sk=2)
        {"geo_sk": 2, "time_sk": sample_time_dim, "condition_sk": 10, "demographic_sk": 101, "count": 3}, # P_Arthritis_0_14
        {"geo_sk": 2, "time_sk": sample_time_dim, "condition_sk": 20, "demographic_sk": 102, "count": 8}, # F_Asthma_15_24
        # M_NS_85_over is 0, so filtered out
        {"geo_sk": 2, "time_sk": sample_time_dim, "condition_sk": 40, "demographic_sk": 105, "count": 89}  # P_No_condition_Tot (mapped to demo_sk 104 or 105 based on join)
    ]
    expected_df = pl.DataFrame(expected_data)
    # Ensure dtypes match the potential output (adjust if necessary)
    expected_df = expected_df.with_columns([
        pl.col("geo_sk").cast(pl.Int64),
        pl.col("time_sk").cast(pl.Int64),
        pl.col("condition_sk").cast(pl.Int64),
        pl.col("demographic_sk").cast(pl.Int64),
        pl.col("count").cast(pl.Int64)
    ])

    # Define the columns expected in the final fact table
    expected_cols = ["geo_sk", "time_sk", "condition_sk", "demographic_sk", "count"]
    assert all(col in result_df.columns for col in expected_cols), \
           f"Result missing expected fact columns. Found: {result_df.columns}"

    # Select only expected columns and sort for comparison
    result_sorted = result_df.select(expected_cols).sort(["geo_sk", "condition_sk", "demographic_sk"])
    expected_sorted = expected_df.select(expected_cols).sort(["geo_sk", "condition_sk", "demographic_sk"])

    assert_frame_equal(result_sorted, expected_sorted)


# TODO: Add integration test for a complex table like G19/G20/G21
# This will require setting up relevant dimension files (e.g., health_condition, demographic)
# and verifying the more complex unpivoted structure and joins.
