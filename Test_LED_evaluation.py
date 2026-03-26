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
import rasterio
from rasterio.features import shapes
from rasterio.mask import mask as rio_mask
from shapely.geometry import shape

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

AGG_RULES = {
    "BUPR": "sum",
    "BUPL": "sum",
    "BUA": "sum",
    "BUI": "mean",
    "FUPY": "mean",
}
# %%
'''Define helper functions for data processing and evaluation.'''

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


def raster_to_polygons(raster_path: Path, clip_boundary: gpd.GeoDataFrame, value_col: str) -> gpd.GeoDataFrame:
    """Convert raster pixels to polygons and keep valid pixel values."""
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        boundary = clip_boundary.to_crs(src.crs)
        data, transform = rio_mask(src, boundary.geometry, crop=True)

        band = data[0]
        nodata = src.nodata
        valid_mask = np.ones(band.shape, dtype=bool) if nodata is None else band != nodata

        records: list[dict[str, object]] = []
        for geom, value in shapes(band, mask=valid_mask, transform=transform):
            val = float(value)
            if np.isnan(val):
                continue
            records.append({"geometry": shape(geom), value_col: val})

    if not records:
        raise ValueError(f"No valid raster cells found in {raster_path.name}")

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=raster_crs)
    return gdf


def get_value_column(dataset_name: str, gdf: gpd.GeoDataFrame) -> str:
    """Pick the main numeric value column for aggregation."""
    preferred = f"{dataset_name}_value"
    if preferred in gdf.columns:
        return preferred

    numeric_cols = [c for c in gdf.columns if c != "geometry" and pd.api.types.is_numeric_dtype(gdf[c])]
    if numeric_cols:
        return numeric_cols[0]

    gdf[f"{dataset_name}_count"] = 1
    return f"{dataset_name}_count"


def aggregate_to_master_grid(
    master_grid: gpd.GeoDataFrame,
    source_gdf: gpd.GeoDataFrame,
    source_col: str,
    out_col: str,
    agg_func: str,
) -> gpd.GeoDataFrame:
    """Spatially join source data to master grid and aggregate by HISDAC_id."""
    source_in_master_crs = source_gdf.to_crs(master_grid.crs)
    joined = gpd.sjoin(
        source_in_master_crs[[source_col, "geometry"]],
        master_grid[["HISDAC_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )

    aggregated = joined.groupby("HISDAC_id")[source_col].agg(agg_func)
    master_grid[out_col] = master_grid["HISDAC_id"].map(aggregated)
    return master_grid


def attach_led_to_grid(master_grid: gpd.GeoDataFrame, led_gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Attach HISDAC_id to LED points and count LED points in each grid cell."""
    led_in_master_crs = led_gdf.to_crs(master_grid.crs)
    led_with_id = gpd.sjoin(
        led_in_master_crs,
        master_grid[["HISDAC_id", "geometry"]],
        how="left",
        predicate="within",
    )

    led_counts = led_with_id.groupby("HISDAC_id").size()
    master_grid["LED_count"] = master_grid["HISDAC_id"].map(led_counts).fillna(0).astype(int)
    return master_grid, led_with_id
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
            gdf = gpd.read_file(data_file)
            hisdac_data[name] = gpd.clip(gdf, wa_boundary.to_crs(gdf.crs))
        else:
            hisdac_data[name] = raster_to_polygons(data_file, wa_boundary, value_col=f"{name}_value")

    if "BUPR" not in hisdac_data:
        raise ValueError("BUPR is required to build the master HISDAC grid.")

    # Build master grid from BUPR and assign unique HISDAC_id.
    master_grid = hisdac_data["BUPR"].copy().reset_index(drop=True)
    master_grid["HISDAC_id"] = np.arange(1, len(master_grid) + 1)

    # Aggregate each HISDAC dataset to the BUPR master grid.
    for name, gdf in hisdac_data.items():
        source_col = get_value_column(name, gdf)
        agg_func = AGG_RULES.get(name, "mean")
        out_col = f"{name}_{agg_func}"

        if name == "BUPR":
            master_grid[out_col] = master_grid[source_col]
            print(f"Assigned {name} -> {out_col} from master grid source values")
            continue

        master_grid = aggregate_to_master_grid(master_grid, gdf, source_col, out_col, agg_func)
        print(f"Aggregated {name} -> {out_col}")

    # Load LED results for Washington
    led_results = gpd.read_file(TESTING_SET_PATH / "WA_LED_points-5070.gpkg")
    master_grid, led_with_id = attach_led_to_grid(master_grid, led_results)
    print("Linked LED points to master grid using HISDAC_id.")

    # Save outputs for downstream LED evaluation.
    master_out = TESTING_SET_PATH / f"{state_code}_{analysis_year}_HISDAC_master_grid.gpkg"
    led_out = TESTING_SET_PATH / f"{state_code}_{analysis_year}_LED_with_HISDAC_id.gpkg"
    master_grid.to_file(master_out, driver="GPKG")
    led_with_id.to_file(led_out, driver="GPKG")
    print(f"Saved master grid: {master_out}")
    print(f"Saved LED joined points: {led_out}")

    # Show quick summary of selected aggregated fields.
    summary_cols = [c for c in master_grid.columns if c.startswith(("BUPR_", "BUPL_", "BUA_", "BUI_", "FUPY_", "LED_count"))]
    print("\nMaster grid summary:")
    print(master_grid[["HISDAC_id"] + summary_cols].head())
except Exception as e:
    print(f"An error occurred: {e}")
    