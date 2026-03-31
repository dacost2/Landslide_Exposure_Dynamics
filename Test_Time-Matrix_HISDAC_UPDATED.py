# %%
'''code created with the help of Gemini Pro, then verfied, modified and expanded by D. Acosta-Reyes.'''
# %% [markdown]
# # HISDAC Time Matrix Construction (Updated)
# This script constructs a spatiotemporal matrix from the HISDAC datasets, focusing on a
# %%

from __future__ import annotations

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import box
import pyarrow


# Define data path
DATA_PATH = Path("/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics/Data")

# Define paths to FILES
HISDAC_PATH = DATA_PATH / "HISDAC_US_V2"
STATE_BOUNDARY_PATH = DATA_PATH / "cb_2024_us_state_500k.zip"
TESTING_SET_PATH = DATA_PATH / "Testing_Set"

HISDAC_DATASETS = {
    "BUPR": HISDAC_PATH / "Historical_Built-up_Records_BUPR_V2",
    "BUPL": HISDAC_PATH / "Historical_Built-up_Property_Locations_BUPL_V2",
    "FBUY": HISDAC_PATH / "Historical_Settlement_Year_Built_Layer_1810-2020_V2"
}

# --- [Keep your existing helper functions: extract_tar_files, get_available_years, choose_year, choose_state_code, find_dataset_file] ---
# (I am omitting them here for brevity, but keep them in your script!)

'''Define helper functions for data processing and evaluation.'''


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


def choose_state_code(state_boundaries: gpd.GeoDataFrame, default_state: str = "WA") -> str:
    """Ask user to choose a state code from available STUSPS values."""
    if "STUSPS" not in state_boundaries.columns:
        raise ValueError("State boundary file is missing STUSPS column.")

    state_codes = sorted(state_boundaries["STUSPS"].dropna().astype(str).unique().tolist())
    if not state_codes:
        raise ValueError("No state codes found in state boundary dataset.")

    print("\nAvailable state codes (examples):")
    print(", ".join(state_codes[:20]) + (" ..." if len(state_codes) > 20 else ""))
    print(f"Default state: {default_state}")

    while True:
        selected = input("Choose state code (press Enter for default): ").strip().upper()
        if selected == "":
            selected = default_state
        if selected in state_codes:
            return selected
        print("Invalid state code. Please enter a valid STUSPS value (for example WA, OR, CA).")


def find_dataset_file(dataset_name: str, dataset_path: Path, year: str | None = None) -> Path:
    """Find one HISDAC file, optionally filtered by year, preferring shapefile over tif."""
    shp_candidates: list[Path] = []
    tif_candidates: list[Path] = []
    exact_shp: list[Path] = []
    exact_tif: list[Path] = []

    for file_path in sorted(dataset_path.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in {".shp", ".tif"}:
            continue
        stem = file_path.stem.lower()
        if dataset_name.lower() not in stem:
            continue
        if year is not None and not stem.startswith(f"{year}_"):
            continue

        if year is not None and stem == f"{year}_{dataset_name.lower()}":
            if file_path.suffix.lower() == ".shp":
                exact_shp.append(file_path)
            else:
                exact_tif.append(file_path)

        if file_path.suffix.lower() == ".shp":
            shp_candidates.append(file_path)
        else:
            tif_candidates.append(file_path)

    # BUI can unpack to deeply nested internal source paths, so prefer exact year name and shortest path depth.
    if dataset_name.upper() == "BUI":
        if exact_shp:
            return sorted(exact_shp, key=lambda p: (len(p.parts), str(p)))[0]
        if exact_tif:
            return sorted(exact_tif, key=lambda p: (len(p.parts), str(p)))[0]

    if shp_candidates:
        return sorted(shp_candidates, key=lambda p: (len(p.parts), str(p)))[0]
    if tif_candidates:
        return sorted(tif_candidates, key=lambda p: (len(p.parts), str(p)))[0]

    if year is None:
        raise FileNotFoundError(f"No file found for {dataset_name}")
    raise FileNotFoundError(f"No file found for {dataset_name}, year {year}")

# New functions

def get_clipped_array(raster_path: Path, clip_boundary: gpd.GeoDataFrame) -> tuple[np.ndarray, object]:
    """Reads a raster, clips it to the boundary, and returns the 2D array and transform."""
    with rasterio.open(raster_path) as src:
        boundary = clip_boundary.to_crs(src.crs)
        data, transform = rio_mask(src, boundary.geometry, crop=True)
        band = data[0].astype(np.float32)
        
        # Handle NoData
        if src.nodata is not None:
            band[band == src.nodata] = np.nan
            
        return band, transform

def build_spatial_anchor(base_array: np.ndarray, transform: object, raster_crs: object) -> tuple[gpd.GeoDataFrame, np.ndarray, np.ndarray]:
    """Creates the geometric polygons for valid pixels and returns row/col indices for fast array sampling."""
    # Find rows and columns where data exists (not NaN and > 0)
    valid_mask = (~np.isnan(base_array)) & (base_array > 0)
    rows, cols = np.where(valid_mask)
    
    # Calculate bounding boxes mathematically (100x faster than rasterio.features.shapes)
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
    """
    Map year to end-of-semi-decade label (e.g., 1996-2000 -> 2000).
    Explicitly accumulates all structures built prior to the baseline_year 
    into a single baseline bin to establish the initial built environment.
    """
    y = pd.to_numeric(series, errors="coerce")
    
    # 1. Calculate the standard 5-year bins
    semi_decade = ((y - 1) // 5 + 1) * 5
    
    # 2. Accumulate everything prior to the baseline
    semi_decade = np.where(semi_decade < baseline_year, baseline_year, semi_decade)
    
    # 3. Cap future outliers (e.g., a typo like "2050" in the LED)
    semi_decade = np.where(semi_decade > max_year, max_year, semi_decade)
    
    return pd.Series(semi_decade, index=series.index)


# %%
# --- MAIN EXECUTION BLOCK ---
try:
    default_state_code = "WA"
    
    # 1. Setup and Boundaries
    state_boundaries = gpd.read_file(STATE_BOUNDARY_PATH)
    state_code = choose_state_code(state_boundaries, default_state=default_state_code)
    wa_boundary = state_boundaries[state_boundaries["STUSPS"] == state_code]
    print(f"Selected state: {state_code}")

    # 2. Establish the Spatial Anchor using BUPL 2020 as the base grid
    bupl_2020_file = find_dataset_file("BUPL", HISDAC_DATASETS["BUPL"], "2020")
    print("\n1) Generating Spatial Anchor from BUPL 2020...")
    base_array, transform = get_clipped_array(bupl_2020_file, wa_boundary)
    
    with rasterio.open(bupl_2020_file) as src:
        raster_crs = src.crs

    # spatial_anchor is your GDF for joining with LED. rows/cols are for instant data extraction.
    spatial_anchor, valid_rows, valid_cols = build_spatial_anchor(base_array, transform, raster_crs)
    print(f"   -> Created {len(spatial_anchor)} valid pixel polygons.")

    # 3. Initialize the Matrix Dictionary
    matrix_data = {"HISDAC_id": spatial_anchor["HISDAC_id"].values}

    # 4. Extract FBUY
    print("2) Extracting FBUY...")
    fbuy_file = find_dataset_file("FBUY", HISDAC_DATASETS["FBUY"])
    fbuy_array, _ = get_clipped_array(fbuy_file, wa_boundary)
    matrix_data["FBUY"] = fbuy_array[valid_rows, valid_cols]

    # 5. Extract and Calculate Density (BUPR_2020 / BUPL_2020)
    print("3) Calculating Structural Density...")
    bupr_2020_file = find_dataset_file("BUPR", HISDAC_DATASETS["BUPR"], "2020")
    bupr_array, _ = get_clipped_array(bupr_2020_file, wa_boundary)
    
    bupr_vals = bupr_array[valid_rows, valid_cols]
    bupl_vals = base_array[valid_rows, valid_cols] # We already loaded BUPL 2020
    
    # Avoid division by zero
    density = np.where(bupl_vals > 0, bupr_vals / bupl_vals, 0)
    matrix_data["DENSITY"] = density

    # 6. Time-Series Delta Processing
    print("4) Processing BUPL Time Series Deltas...")
    bupl_years = get_available_years(HISDAC_DATASETS["BUPL"], "BUPL")
    # Filter for your desired decades (e.g., 1915 to 2020)
    target_years = sorted([y for y in bupl_years if 1915 <= int(y) <= 2020])
    
    # NEW: Extract BUPR Time-Series Deltas for Tier 4 High-Rise Logic
    print("4b) Processing BUPR Time Series Deltas...")
    bupr_years = get_available_years(HISDAC_DATASETS["BUPR"], "BUPR")
    target_bupr_years = sorted([y for y in bupr_years if 1915 <= int(y) <= 2020])
    
    bupr_time_arrays = {}
    for year in target_bupr_years:
        file_path = find_dataset_file("BUPR", HISDAC_DATASETS["BUPR"], year)
        arr, _ = get_clipped_array(file_path, wa_boundary)
        bupr_time_arrays[year] = arr[valid_rows, valid_cols]

    for i in range(1, len(target_bupr_years)):
        prev_yr = target_bupr_years[i-1]
        curr_yr = target_bupr_years[i]
        delta = bupr_time_arrays[curr_yr] - bupr_time_arrays[prev_yr]
        delta = np.where(delta < 0, 0, delta) 
        matrix_data[f"D_BUPR{curr_yr}"] = delta
        
    matrix_data[f"D_BUPR{target_bupr_years[0]}"] = bupr_time_arrays[target_bupr_years[0]]
    
    # Dictionary to hold the raw arrays temporarily
    bupl_time_arrays = {}
    for year in target_years:
        file_path = find_dataset_file("BUPL", HISDAC_DATASETS["BUPL"], year)
        arr, _ = get_clipped_array(file_path, wa_boundary)
        bupl_time_arrays[year] = arr[valid_rows, valid_cols]

    # Calculate Deltas
    for i in range(1, len(target_years)):
        prev_yr = target_years[i-1]
        curr_yr = target_years[i]
        
        # Calculate new units built in this 5-year window
        delta = bupl_time_arrays[curr_yr] - bupl_time_arrays[prev_yr]
        
        # Clean up negative anomalies (sometimes present in historical data)
        delta = np.where(delta < 0, 0, delta) 
        matrix_data[f"D_BUPL{curr_yr}"] = delta
        
    # If you want 1915/1920 to represent "everything built up to that point", you can keep it:
    matrix_data[f"D_BUPL{target_years[0]}"] = bupl_time_arrays[target_years[0]]

    # 7. Assemble and Export
    print("\n5) Assembling Matrix and Exporting...")
    wide_matrix_df = pd.DataFrame(matrix_data)
    
    # Save the spatial anchor (The "Where")
    spatial_out = TESTING_SET_PATH / f"{state_code}_HISDAC_Spatial_Anchor.gpkg"
    spatial_anchor.to_file(spatial_out, driver="GPKG")
    
    # Save the Wide Matrix (The "When" and "What") as Parquet for speed
    matrix_out = TESTING_SET_PATH / f"{state_code}_HISDAC_Wide_Matrix.parquet"
    wide_matrix_df.to_parquet(matrix_out, index=False)

    print(f"SUCCESS!")
    print(f"-> Spatial Anchor saved to: {spatial_out}")
    print(f"-> Wide Matrix saved to: {matrix_out}")
    
    print("\nWide Matrix Preview:")
    print(wide_matrix_df.head())

except Exception as e:
    print(f"An error occurred: {e}")


# %%
# Apply it to the LED database

# --- 1. Define led_joined (The Spatial Link) ---
print("1) Loading Spatial Anchor and LED Inventory...")
spatial_anchor_path = TESTING_SET_PATH / f"{state_code}_HISDAC_Spatial_Anchor.gpkg"
led_file_path = TESTING_SET_PATH / f"{state_code}_LED_points-5070.gpkg"

spatial_anchor = gpd.read_file(spatial_anchor_path)
led_points = gpd.read_file(led_file_path)

print("2) Spatially joining LED buildings to HISDAC grid...")
# Ensure CRS matches before joining
if led_points.crs != spatial_anchor.crs:
    led_points = led_points.to_crs(spatial_anchor.crs)

# Perform the spatial join. We use 'within' assuming LED points are centroids.
led_joined = gpd.sjoin(
    led_points, 
    spatial_anchor[["HISDAC_id", "geometry"]], 
    how="inner", 
    predicate="within"
)

# Apply the semi-decade accumulation logic
led_joined["semi_decade"] = semidecade_from_med_yr_blt(led_joined["med_yr_blt"])

# --- 2. Define led_wide (The Tabular Pivot) ---
print("3) Pivoting LED inventory into Wide Matrix format...")

# Crosstab counts the number of buildings per semi_decade for each HISDAC_id
led_wide = pd.crosstab(
    index=led_joined["HISDAC_id"], 
    columns=led_joined["semi_decade"],
    dropna=False
)

# Rename the columns to explicitly match our target format (e.g., "LED_1920")
# We cast col to int to drop the '.0' float formatting that sometimes occurs
led_wide.columns = [f"LED_{int(col)}" for col in led_wide.columns]

# Reset the index so HISDAC_id becomes a standard column for merging
led_wide = led_wide.reset_index()

# Ensure all expected columns exist (1920 to 2020, step 5), in case some years had zero buildings statewide
expected_years = range(1920, 2025, 5)
for year in expected_years:
    col_name = f"LED_{year}"
    if col_name not in led_wide.columns:
        led_wide[col_name] = 0

# Sort columns to keep the matrix organized (HISDAC_id first, then chronologically)
col_order = ["HISDAC_id"] + [f"LED_{y}" for y in expected_years]
led_wide = led_wide[col_order]

print("\nled_wide preview:")
print(led_wide.head())

# --- 3. Create the Master Matrix ---
print("\n4) Merging with HISDAC Matrix...")
hisdac_matrix = pd.read_parquet(TESTING_SET_PATH / f"{state_code}_HISDAC_Wide_Matrix.parquet")

master_matrix = hisdac_matrix.merge(led_wide, on="HISDAC_id", how="left")

# Any pixel that exists in HISDAC but has NO buildings in LED will have NaNs for the LED columns. Fill with 0.
led_cols = [col for col in master_matrix.columns if col.startswith("LED_")]
master_matrix[led_cols] = master_matrix[led_cols].fillna(0).astype(int)

# Save the final consolidated matrix
master_out_path = TESTING_SET_PATH / f"{state_code}_Master_Spatiotemporal_Matrix.parquet"
master_matrix.to_parquet(master_out_path, index=False)

# ... [Your existing code: master_matrix.to_parquet(...)] ...

# NEW: Save the building-level inventory with the attached HISDAC_id
print("5) Saving Building-Level Inventory...")
led_joined_out_path = TESTING_SET_PATH / f"{state_code}_LED_Joined_Buildings.parquet"

# Drop geometry for lightning-fast Parquet tabular storage. 
# You can always rejoin it to your spatial LED file later via the LED ID.
led_tabular = pd.DataFrame(led_joined.drop(columns='geometry'))
led_tabular.to_parquet(led_joined_out_path, index=False)

print(f"SUCCESS! Master Matrix saved to: {master_out_path}")
print(f"SUCCESS! Joined Buildings saved to: {led_joined_out_path}")