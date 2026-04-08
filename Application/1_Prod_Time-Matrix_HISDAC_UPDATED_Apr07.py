# %% [markdown]
# # HISDAC Time Matrix Construction (Production Optimized)
# #### Daniel Acosta-Reyes
# April 07, 2026
# %%
from __future__ import annotations

import pandas as pd
import geopandas as gpd
import numpy as np
import re
import argparse
from pathlib import Path
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import box

# --- 1. Argparse Setup for Master Controller ---
parser = argparse.ArgumentParser(description="Process HISDAC Matrices for a specific state.")
parser.add_argument("--state", type=str, required=True, help="2-letter state code (e.g., WA)")
# Use parse_known_args to play nicely if running inside an interactive Jupyter environment by mistake
args, unknown = parser.parse_known_args()
STATE_CODE = args.state.upper()

# --- 2. Paths & State Dictionary ---
DATA_PATH = Path("/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics/Data")
HISDAC_PATH = DATA_PATH / "HISDAC_US_V2"
STATE_BOUNDARY_PATH = DATA_PATH / "cb_2024_us_state_500k.zip"
BUILDING_INVENTORY_PATH = DATA_PATH / "LED_by_State_GPKG"

# Create state-specific production directory
PRODUCTION_SET_PATH = DATA_PATH / "Production_set" / STATE_CODE
PRODUCTION_SET_PATH.mkdir(parents=True, exist_ok=True)

HISDAC_DATASETS = {
    "BUPR": HISDAC_PATH / "Historical_Built-up_Records_BUPR_V2",
    "BUPL": HISDAC_PATH / "Historical_Built-up_Property_Locations_BUPL_V2",
    "FBUY": HISDAC_PATH / "Historical_Settlement_Year_Built_Layer_1810-2020_V2"
}

state_dict = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"
}

STATE_NAME = state_dict.get(STATE_CODE)
if not STATE_NAME:
    raise ValueError(f"Invalid state code provided: {STATE_CODE}")

# --- 3. Optimized Helper Functions ---

# OPTIMIZATION: Cache the file paths so we don't rglob the entire disk inside a loop
FILE_CACHE = {}

def index_dataset_files(dataset_path: Path):
    """Scans a dataset directory ONCE and caches the file paths."""
    if dataset_path not in FILE_CACHE:
        FILE_CACHE[dataset_path] = list(dataset_path.rglob("*.[st][hi][pf]*")) # Matches .shp and .tif
    return FILE_CACHE[dataset_path]

def get_available_years(dataset_path: Path, dataset_name: str) -> list[str]:
    year_pattern = re.compile(r"^(\d{4})_")
    years: set[str] = set()
    all_files = index_dataset_files(dataset_path)

    for file_path in all_files:
        if dataset_name.lower() in file_path.stem.lower():
            match = year_pattern.match(file_path.stem)
            if match:
                years.add(match.group(1))
    return sorted(years)

def find_dataset_file(dataset_name: str, dataset_path: Path, year: str | None = None) -> Path:
    all_files = index_dataset_files(dataset_path)
    candidates = []

    for file_path in all_files:
        stem = file_path.stem.lower()
        if dataset_name.lower() not in stem:
            continue
        if year is not None and not stem.startswith(f"{year}_"):
            continue
        candidates.append(file_path)

    if not candidates:
        raise FileNotFoundError(f"No file found for {dataset_name}, year {year}")
        
    # Prefer exact names and .shp, sort by path depth
    candidates.sort(key=lambda p: (len(p.parts), 0 if p.suffix.lower() == '.shp' else 1, str(p)))
    return candidates[0]

def get_clipped_array(raster_path: Path, clip_boundary: gpd.GeoDataFrame) -> tuple[np.ndarray, object]:
    with rasterio.open(raster_path) as src:
        boundary = clip_boundary.to_crs(src.crs)
        data, transform = rio_mask(src, boundary.geometry, crop=True)
        band = data[0].astype(np.float32)
        if src.nodata is not None:
            band[band == src.nodata] = np.nan
        return band, transform

def build_spatial_anchor(base_array: np.ndarray, transform: object, raster_crs: object) -> tuple[gpd.GeoDataFrame, np.ndarray, np.ndarray]:
    valid_mask = (~np.isnan(base_array)) & (base_array > 0)
    rows, cols = np.where(valid_mask)
    
    records = []
    for row, col in zip(rows, cols):
        x_left, y_top = rasterio.transform.xy(transform, row, col, offset="ul")
        x_right, y_bottom = rasterio.transform.xy(transform, row, col, offset="lr")
        geom = box(min(x_left, x_right), min(y_bottom, y_top), max(x_left, x_right), max(y_bottom, y_top))
        records.append(geom)

    gdf = gpd.GeoDataFrame({"geometry": records}, crs=raster_crs)
    gdf["HISDAC_id"] = np.arange(1, len(gdf) + 1)
    return gdf, rows, cols

def semidecade_from_med_yr_blt(series: pd.Series, baseline_year: int = 1920, max_year: int = 2020) -> pd.Series:
    y = pd.to_numeric(series, errors="coerce")
    semi_decade = ((y - 1) // 5 + 1) * 5
    semi_decade = np.where(semi_decade < baseline_year, baseline_year, semi_decade)
    semi_decade = np.where(semi_decade > max_year, max_year, semi_decade)
    return pd.Series(semi_decade, index=series.index)


# --- 4. Main Execution Block ---
def main():
    print(f"=== Processing {STATE_NAME} ({STATE_CODE}) ===")
    
    # 1. Boundaries
    state_boundaries = gpd.read_file(STATE_BOUNDARY_PATH)
    wa_boundary = state_boundaries[state_boundaries["STUSPS"] == STATE_CODE]

    # 2. Establish Spatial Anchor
    print("1) Generating Spatial Anchor from BUPL 2020...")
    bupl_2020_file = find_dataset_file("BUPL", HISDAC_DATASETS["BUPL"], "2020")
    base_array, transform = get_clipped_array(bupl_2020_file, wa_boundary)
    
    with rasterio.open(bupl_2020_file) as src:
        raster_crs = src.crs

    spatial_anchor, valid_rows, valid_cols = build_spatial_anchor(base_array, transform, raster_crs)
    matrix_data = {"HISDAC_id": spatial_anchor["HISDAC_id"].values}

    # 3. Extract FBUY
    print("2) Extracting FBUY...")
    fbuy_file = find_dataset_file("FBUY", HISDAC_DATASETS["FBUY"])
    fbuy_array, _ = get_clipped_array(fbuy_file, wa_boundary)
    matrix_data["FBUY"] = fbuy_array[valid_rows, valid_cols]

    # 4. Extract Density
    print("3) Calculating Structural Density...")
    bupr_2020_file = find_dataset_file("BUPR", HISDAC_DATASETS["BUPR"], "2020")
    bupr_array, _ = get_clipped_array(bupr_2020_file, wa_boundary)
    
    bupr_vals = bupr_array[valid_rows, valid_cols]
    bupl_vals = base_array[valid_rows, valid_cols] 
    density = np.where(bupl_vals > 0, bupr_vals / bupl_vals, 0)
    matrix_data["DENSITY"] = density

    # 5. Time-Series Delta Processing
    print("4) Processing BUPR and BUPL Time Series Deltas...")
    bupl_years = get_available_years(HISDAC_DATASETS["BUPL"], "BUPL")
    target_years = sorted([y for y in bupl_years if 1915 <= int(y) <= 2020])
    
    bupr_years = get_available_years(HISDAC_DATASETS["BUPR"], "BUPR")
    target_bupr_years = sorted([y for y in bupr_years if 1915 <= int(y) <= 2020])
    
    # Process BUPR Deltas
    bupr_time_arrays = {}
    for year in target_bupr_years:
        file_path = find_dataset_file("BUPR", HISDAC_DATASETS["BUPR"], year)
        arr, _ = get_clipped_array(file_path, wa_boundary)
        bupr_time_arrays[year] = arr[valid_rows, valid_cols]

    for i in range(1, len(target_bupr_years)):
        prev_yr, curr_yr = target_bupr_years[i-1], target_bupr_years[i]
        delta = np.maximum(0, bupr_time_arrays[curr_yr] - bupr_time_arrays[prev_yr])
        matrix_data[f"D_BUPR{curr_yr}"] = delta
    matrix_data[f"D_BUPR{target_bupr_years[0]}"] = bupr_time_arrays[target_bupr_years[0]]
    
    # Process BUPL Deltas
    bupl_time_arrays = {}
    for year in target_years:
        file_path = find_dataset_file("BUPL", HISDAC_DATASETS["BUPL"], year)
        arr, _ = get_clipped_array(file_path, wa_boundary)
        bupl_time_arrays[year] = arr[valid_rows, valid_cols]

    for i in range(1, len(target_years)):
        prev_yr, curr_yr = target_years[i-1], target_years[i]
        delta = np.maximum(0, bupl_time_arrays[curr_yr] - bupl_time_arrays[prev_yr])
        matrix_data[f"D_BUPL{curr_yr}"] = delta
    matrix_data[f"D_BUPL{target_years[0]}"] = bupl_time_arrays[target_years[0]]

    # 6. Assemble and Export HISDAC Matrix
    print("5) Assembling Matrix and Exporting to State Folder...")
    wide_matrix_df = pd.DataFrame(matrix_data)
    
    spatial_out = PRODUCTION_SET_PATH / f"{STATE_CODE}_HISDAC_Spatial_Anchor.gpkg"
    spatial_anchor.to_file(spatial_out, driver="GPKG")
    
    matrix_out = PRODUCTION_SET_PATH / f"{STATE_CODE}_HISDAC_Wide_Matrix.parquet"
    wide_matrix_df.to_parquet(matrix_out, index=False)

    # 7. LED Integration
    print("6) Loading Spatial Anchor and LED Inventory...")
    led_file_path = BUILDING_INVENTORY_PATH / f"{STATE_NAME}_LED.gpkg"
    led_points = gpd.read_file(led_file_path)

    if led_points.crs != spatial_anchor.crs:
        led_points = led_points.to_crs(spatial_anchor.crs)

    led_joined = gpd.sjoin(led_points, spatial_anchor[["HISDAC_id", "geometry"]], how="inner", predicate="within")
    led_joined["semi_decade"] = semidecade_from_med_yr_blt(led_joined["med_yr_blt"])

    print("7) Pivoting LED inventory into Wide Matrix format...")
    led_wide = pd.crosstab(index=led_joined["HISDAC_id"], columns=led_joined["semi_decade"], dropna=False)
    led_wide.columns = [f"LED_{int(col)}" for col in led_wide.columns]
    led_wide = led_wide.reset_index()

    expected_years = range(1920, 2025, 5)
    for year in expected_years:
        if f"LED_{year}" not in led_wide.columns:
            led_wide[f"LED_{year}"] = 0

    col_order = ["HISDAC_id"] + [f"LED_{y}" for y in expected_years]
    led_wide = led_wide[col_order]

    print("8) Merging with HISDAC Matrix...")
    master_matrix = wide_matrix_df.merge(led_wide, on="HISDAC_id", how="left")
    led_cols = [col for col in master_matrix.columns if col.startswith("LED_")]
    master_matrix[led_cols] = master_matrix[led_cols].fillna(0).astype(int)

    # 8. Final Exports
    print("9) Saving Final Production Files...")
    master_out_path = PRODUCTION_SET_PATH / f"{STATE_CODE}_Master_Spatiotemporal_Matrix.parquet"
    master_matrix.to_parquet(master_out_path, index=False)

    # Export GPKG and Parquet for the joined building inventory
    led_joined_gpkg = PRODUCTION_SET_PATH / f"{STATE_CODE}_LED_Joined_Buildings.gpkg"
    led_joined_parquet = PRODUCTION_SET_PATH / f"{STATE_CODE}_LED_Joined_Buildings.parquet"
    
    led_joined.to_file(led_joined_gpkg, driver="GPKG")
    led_joined.drop(columns='geometry').to_parquet(led_joined_parquet, index=False)

    print(f"=== {STATE_CODE} SCRIPT 1 COMPLETE ===")

if __name__ == "__main__":
    main()