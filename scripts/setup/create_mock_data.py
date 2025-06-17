#!/usr/bin/env python3
"""Create comprehensive mock data for testing the AHGD ETL pipeline."""

import polars as pl
import geopandas as gpd
from shapely.geometry import Polygon
import random
from pathlib import Path
import zipfile


def create_mock_geographic_data():
    """Create mock shapefiles for all geographic levels."""
    print("Creating mock geographic data...")
    
    levels = {
        "sa1": 50,   # 50 SA1s
        "sa2": 20,   # 20 SA2s
        "sa3": 10,   # 10 SA3s
        "sa4": 5,    # 5 SA4s
        "ste": 8,    # 8 States/Territories
    }
    
    for level, count in levels.items():
        print(f"  Creating {level.upper()}: {count} areas")
        
        # Create data
        data = []
        for i in range(count):
            # Random location around Australia
            lon = 115 + random.uniform(0, 35)  # 115-150 E
            lat = -40 + random.uniform(0, 30)   # -40 to -10 S
            
            # Small polygon
            polygon = Polygon([
                (lon, lat),
                (lon + 0.1, lat),
                (lon + 0.1, lat + 0.1),
                (lon, lat + 0.1),
                (lon, lat)
            ])
            
            # State assignment
            state_code = str((i % 8) + 1)
            state_names = ["NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT"]
            
            record = {
                f"{level.upper()}_CODE_2": f"{level.upper()}{10000 + i}",
                f"{level.upper()}_NAME_2": f"Test {level.upper()} {i}",
                "STE_CODE_2": state_code,
                "STE_NAME_2": state_names[int(state_code) - 1],
                "geometry": polygon
            }
            data.append(record)
        
        # Create GeoDataFrame and save
        gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
        output_dir = Path(f"data/raw/geographic/{level}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        shapefile_path = output_dir / f"mock_{level}_2021.shp"
        gdf.to_file(shapefile_path)
        
        # Create zip file
        zip_path = Path(f"data/raw/geographic/1270055001_{level}_2021_aust_shape.zip")
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                file_path = output_dir / f"mock_{level}_2021{ext}"
                if file_path.exists():
                    zf.write(file_path, file_path.name)


def create_mock_census_data():
    """Create mock census CSV files."""
    print("\nCreating mock census data...")
    
    # Use 100 SA1s for census data
    num_sa1s = 100
    sa1_codes = [f"1{random.randint(1000000000, 1999999999)}" for _ in range(num_sa1s)]
    
    # G01 - Population
    print("  Creating G01 (Population)...")
    g01_data = {
        "SA1_CODE_2021": sa1_codes,
        "Tot_P_M": [random.randint(200, 800) for _ in range(num_sa1s)],
        "Tot_P_F": [random.randint(200, 800) for _ in range(num_sa1s)],
    }
    g01_data["Tot_P_P"] = [m + f for m, f in zip(g01_data["Tot_P_M"], g01_data["Tot_P_F"])]
    
    g01_dir = Path("data/raw/census/extracted/2021Census_G01_AUS")
    g01_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(g01_data).write_csv(g01_dir / "2021Census_G01_AUS_SA1.csv")
    
    # G17A - Income
    print("  Creating G17A (Income)...")
    g17_data = {"SA1_CODE_2021": sa1_codes}
    income_brackets = [
        "Neg_Nil_income", "1_149", "150_299", "300_399", "400_499",
        "500_649", "650_799", "800_999", "1000_1249", "1250_1499",
        "1500_1749", "1750_1999", "2000_2999", "3000_more"
    ]
    
    for bracket in income_brackets:
        g17_data[f"P_{bracket}_Tot"] = [random.randint(10, 200) for _ in range(num_sa1s)]
    
    g17_dir = Path("data/raw/census/extracted/2021Census_G17A_AUS")
    g17_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(g17_data).write_csv(g17_dir / "2021Census_G17A_AUS_SA1.csv")
    
    # G18 - Assistance needed
    print("  Creating G18 (Assistance needed)...")
    g18_data = {
        "SA1_CODE_2021": sa1_codes,
        "P_Tot_Need_for_assistance": [random.randint(20, 100) for _ in range(num_sa1s)],
        "P_Tot_No_need_for_assistance": [random.randint(500, 1200) for _ in range(num_sa1s)],
        "P_Tot_Need_not_stated": [random.randint(10, 50) for _ in range(num_sa1s)],
    }
    
    g18_dir = Path("data/raw/census/extracted/2021Census_G18_AUS")
    g18_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(g18_data).write_csv(g18_dir / "2021Census_G18_AUS_SA1.csv")
    
    # G19 - Health conditions
    print("  Creating G19 (Health conditions)...")
    g19_data = {"SA1_CODE_2021": sa1_codes}
    conditions = [
        "Arthritis", "Asthma", "Cancer", "Dementia", "Diabetes",
        "Heart_disease", "Kidney_disease", "Lung_condition",
        "Mental_health_condition", "Stroke", "Other"
    ]
    
    for condition in conditions:
        g19_data[f"P_{condition}_Tot"] = [random.randint(5, 150) for _ in range(num_sa1s)]
    
    g19_dir = Path("data/raw/census/extracted/2021Census_G19_AUS")
    g19_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(g19_data).write_csv(g19_dir / "2021Census_G19_AUS_SA1.csv")
    
    # G20 - Condition counts
    print("  Creating G20 (Condition counts)...")
    g20_data = {"SA1_CODE_2021": sa1_codes}
    for i in range(7):
        col_name = f"P_{i}_Cond_Tot" if i < 6 else "P_6_more_Cond_Tot"
        g20_data[col_name] = [random.randint(50, 300) for _ in range(num_sa1s)]
    
    g20_dir = Path("data/raw/census/extracted/2021Census_G20_AUS")
    g20_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(g20_data).write_csv(g20_dir / "2021Census_G20_AUS_SA1.csv")
    
    # G25 - Unpaid assistance
    print("  Creating G25 (Unpaid assistance)...")
    g25_data = {
        "SA1_CODE_2021": sa1_codes,
        "P_Tot_prvided_unpaid_assist": [random.randint(100, 400) for _ in range(num_sa1s)],
        "P_Tot_no_unpaid_assist": [random.randint(400, 900) for _ in range(num_sa1s)],
        "P_Tot_unpaid_assist_ns": [random.randint(20, 100) for _ in range(num_sa1s)],
    }
    
    g25_dir = Path("data/raw/census/extracted/2021Census_G25_AUS")
    g25_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(g25_data).write_csv(g25_dir / "2021Census_G25_AUS_SA1.csv")


def main():
    """Create all mock data."""
    print("AHGD ETL Mock Data Generator")
    print("=" * 50)
    
    # Create directories
    Path("data/raw/geographic").mkdir(parents=True, exist_ok=True)
    Path("data/raw/census").mkdir(parents=True, exist_ok=True)
    
    # Generate data
    create_mock_geographic_data()
    create_mock_census_data()
    
    print("\n✅ Mock data generation complete!")
    print("\nGenerated files:")
    print("- Geographic: 5 levels (SA1-SA4, STE) with shapefiles")
    print("- Census: G01, G17A, G18, G19, G20, G25 with 100 SA1s each")
    print("\nYou can now run the ETL pipeline with this test data!")


if __name__ == "__main__":
    main()