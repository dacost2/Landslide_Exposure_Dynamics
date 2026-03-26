# %% [markdown]
# # Using HISDAC-US Version II to evaluate Landslide Exposure Database (LED) results
# ##### Author: D. Acosta-Reyes
# ##### University of Washington, Department of Civil and Environmental Engineering 
# ##### Date: 2026-03-26
# ##### Supervisor: Dr. J. Wartman  
#
# This is a test script to evaluate the results of the Landslide Exposure Database (LED) using the HISDAC-US Version II datasets. The script will compare LED outputs with HISDAC-US data to assess the accuracy and reliability of the LED results.
# 
#
# Features:
# - Select state to evaluate (in this case, Washington)
# - Load State of Washington HISDAC-US datasets using auxiliary state boundary files
# - Clip HISDAC-US data to the state boundary. First using 2020 records
# - Compare LED results with HISDAC-US data for the state of Washington
# - Generate evaluation metrics and visualizations to assess LED performance
# %%
from __future__ import annotations

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import tarfile
import re
from pathlib import Path

# Define data path
DATA_PATH = Path("/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics/Data")

# Define paths to FILES
HISDAC_PATH = DATA_PATH / "HISDAC_US_V2"
STATE_BOUNDARY_PATH = DATA_PATH / "cb_2024_us_state_500k.zip"
TESTING_SET_PATH = DATA_PATH / "Testing_Set"

# List of HISDAC-US datasets to evaluate
HISDAC_DATASETS = {
    "BUPR": HISDAC_PATH / "Historical_Built-up_Records_BUPR_V2",
    "BUPL": HISDAC_PATH / "Historical_Built-up_Property_Locations_BUPL_V2",
    "BUA": HISDAC_PATH / "Historical_Built-up_Areas_BUA_V2",
    "BUI": HISDAC_PATH / "Historical_Built-up_Intensity_Layer_BUI_V2",
    "FUPY": HISDAC_PATH / "Historical_Settlement_Year_Built_Layer_1810-2020_V2"
}


def extract_tar_files(dataset_path: Path) -> None:
    """Extract all .tar files in a dataset folder if present."""
    tar_files = sorted(dataset_path.glob("*.tar"))
    if not tar_files:
        return

    for tar_path in tar_files:
        with tarfile.open(tar_path) as tar:
            members = tar.getmembers()
            if not members:
                continue

            sample_target = dataset_path / members[0].name
            if sample_target.exists():
                print(f"Skipping extract (already exists): {tar_path.name}")
                continue

            print(f"Extracting: {tar_path.name}")
            tar.extractall(path=dataset_path)


def get_available_years(dataset_path: Path, dataset_name: str) -> list[str]:
    """Find available years from filenames like 2020_BUPR.tif or 2020_BUPR.shp."""
    year_pattern = re.compile(r"^(\d{4})_")
    years: set[str] = set()

    for file_path in dataset_path.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in {".tif", ".shp"}:
            continue
        if dataset_name.lower() not in file_path.stem.lower():
            continue

        match = year_pattern.match(file_path.stem)
        if match:
            years.add(match.group(1))

    return sorted(years)


def choose_year(years: list[str]) -> str:
    """Ask user to choose one year from available options."""
    if not years:
        raise ValueError("No year folders found after extraction.")

    print("\nAvailable years:")
    print(", ".join(years))

    while True:
        selected = input("Choose year for analysis (for example 2020): ").strip()
        if selected in years:
            return selected
        print("Invalid year. Please pick one of the listed years.")


def find_dataset_file(dataset_name: str, dataset_path: Path, year: str) -> Path:
    """Find one HISDAC file for the selected year, preferring shapefile over tif."""
    shp_candidates: list[Path] = []
    tif_candidates: list[Path] = []

    for file_path in sorted(dataset_path.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in {".shp", ".tif"}:
            continue
        stem = file_path.stem.lower()
        if dataset_name.lower() not in stem:
            continue
        if not stem.startswith(f"{year}_"):
            continue

        if file_path.suffix.lower() == ".shp":
            shp_candidates.append(file_path)
        else:
            tif_candidates.append(file_path)

    if shp_candidates:
        return shp_candidates[0]
    if tif_candidates:
        return tif_candidates[0]

    raise FileNotFoundError(f"No file found for {dataset_name}, year {year}")
# %%
try:
    state_code = "WA"

    # Load state boundary data
    state_boundaries = gpd.read_file(STATE_BOUNDARY_PATH)
    wa_boundary = state_boundaries[state_boundaries["STUSPS"] == state_code]

    # Extract .tar files in each dataset folder (if any), then discover available years
    year_options = None
    for dataset_name, path in HISDAC_DATASETS.items():
        extract_tar_files(path)
        dataset_years = get_available_years(path, dataset_name)
        if not dataset_years:
            continue
        year_options = dataset_years if year_options is None else sorted(set(year_options).intersection(dataset_years))

    if not year_options:
        raise ValueError("No common years found across datasets after extraction.")

    analysis_year = choose_year(year_options)
    print(f"\nSelected year: {analysis_year}")

    # Load HISDAC-US datasets for selected year
    hisdac_data = {}
    for name, path in HISDAC_DATASETS.items():
        data_file = find_dataset_file(name, path, analysis_year)
        print(f"Found {name}: {data_file}")

        if data_file.suffix.lower() == ".shp":
            hisdac_data[name] = gpd.read_file(data_file)
        else:
            # For now, raster (.tif) files are listed and skipped in vector-only workflow.
            print(f"Skipping {name}: {data_file.name} is raster (.tif), not vector (.shp).")

    # Clip loaded vector HISDAC-US data to Washington boundary
    clipped_hisdac_data = {}
    for name, gdf in hisdac_data.items():
        clipped_hisdac_data[name] = gpd.clip(gdf, wa_boundary)

    # Load LED results for Washington
    led_results = gpd.read_file(TESTING_SET_PATH / "LED_WA_Results.shp")

    # Compare LED results with HISDAC-US data
    evaluation_metrics = {}
    for name, gdf in clipped_hisdac_data.items():
        # Example metric: Intersection over Union (IoU) for built-up areas
        if name == "BUA":
            iou = np.sum(led_results.geometry.intersects(gdf.geometry)) / np.sum(led_results.geometry.union(gdf.geometry))
            evaluation_metrics[name] = iou

    # Print evaluation metrics
    print("Evaluation Metrics:")
    for name, metric in evaluation_metrics.items():
        print(f"{name}: {metric:.4f}")
except Exception as e:
    print(f"An error occurred: {e}")

