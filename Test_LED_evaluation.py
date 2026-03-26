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
from shapely.geometry import shape, box

# Define data path
DATA_PATH = Path("/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics/Data")

# Define paths to FILES
HISDAC_PATH = DATA_PATH / "HISDAC_US_V2"
STATE_BOUNDARY_PATH = DATA_PATH / "cb_2024_us_state_500k.zip"
TESTING_SET_PATH = DATA_PATH / "Testing_Set"

# List of HISDAC-US datasets to evaluate
HISDAC_DATASETS = {
    "BUPR": HISDAC_PATH / "Historical_Built-up_Records_BUPR_V2",
    #"BUPL": HISDAC_PATH / "Historical_Built-up_Property_Locations_BUPL_V2",
    #"BUA": HISDAC_PATH / "Historical_Built-up_Areas_BUA_V2",
    "BUI": HISDAC_PATH / "Historical_Built-up_Intensity_Layer_BUI_V2",
    "FBUY": HISDAC_PATH / "Historical_Settlement_Year_Built_Layer_1810-2020_V2"
}

AGG_RULES = {
    "BUPR": "sum",
    "BUPL": "sum",
    "BUA": "sum",
    "BUI": "mean",
    "FBUY": "mean",
}
# %%
'''Define helper functions for data processing and evaluation.'''

def extract_tar_files(dataset_path: Path) -> None:
    """Extract all .tar files in a dataset folder if present."""
    tar_files = sorted(dataset_path.glob("*.tar"))
    if not tar_files:
        return

    for tar_path in tar_files:
        marker = dataset_path / f".{tar_path.stem}.extracted"
        if marker.exists():
            print(f"Skipping extract (marker found): {tar_path.name}")
            continue

        with tarfile.open(tar_path) as tar:
            members = tar.getmembers()
            if not members:
                continue

            print(f"Extracting: {tar_path.name}")
            try:
                tar.extractall(path=dataset_path, filter="data")
            except TypeError:
                # Backward compatibility for Python versions without the filter argument.
                tar.extractall(path=dataset_path)
            marker.touch()


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


def find_dataset_file(dataset_name: str, dataset_path: Path, year: str) -> Path:
    """Find one HISDAC file for the selected year, preferring shapefile over tif."""
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
        if not stem.startswith(f"{year}_"):
            continue

        if stem == f"{year}_{dataset_name.lower()}":
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

    raise FileNotFoundError(f"No file found for {dataset_name}, year {year}")


def raster_to_polygons(raster_path: Path, clip_boundary: gpd.GeoDataFrame, value_col: str) -> gpd.GeoDataFrame:
    """Convert raster pixels to polygons and keep valid pixel values."""
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        boundary = clip_boundary.to_crs(src.crs)
        data, transform = rio_mask(src, boundary.geometry, crop=True)

        band = data[0]
        # rasterio.features.shapes only supports specific dtypes.
        supported_dtypes = {"int16", "int32", "uint8", "uint16", "float32"}
        if str(band.dtype) not in supported_dtypes:
            band = band.astype(np.float32)

        nodata = src.nodata
        if nodata is None:
            valid_mask = np.ones(band.shape, dtype=bool)
        elif isinstance(nodata, float) and np.isnan(nodata):
            valid_mask = ~np.isnan(band)
        else:
            valid_mask = band != nodata

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


def raster_pixels_to_polygons(raster_path: Path, clip_boundary: gpd.GeoDataFrame, value_col: str) -> gpd.GeoDataFrame:
    """Convert each valid raster pixel to one square polygon (no region merging)."""
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        boundary = clip_boundary.to_crs(raster_crs)
        data, transform = rio_mask(src, boundary.geometry, crop=True)

        band = data[0]
        nodata = src.nodata
        if nodata is None:
            valid_mask = np.ones(band.shape, dtype=bool)
        elif isinstance(nodata, float) and np.isnan(nodata):
            valid_mask = ~np.isnan(band)
        else:
            valid_mask = band != nodata

        rows, cols = np.where(valid_mask)
        records: list[dict[str, object]] = []
        for row, col in zip(rows.tolist(), cols.tolist()):
            val = float(band[row, col])
            if np.isnan(val):
                continue

            x_left, y_top = rasterio.transform.xy(transform, row, col, offset="ul")
            x_right, y_bottom = rasterio.transform.xy(transform, row, col, offset="lr")
            geom = box(min(x_left, x_right), min(y_bottom, y_top), max(x_left, x_right), max(y_bottom, y_top))
            records.append({"geometry": geom, value_col: val})

    if not records:
        raise ValueError(f"No valid raster pixels found in {raster_path.name}")

    return gpd.GeoDataFrame(records, geometry="geometry", crs=raster_crs)


def sample_raster_at_grid(raster_path: Path, grid_gdf: gpd.GeoDataFrame, output_col: str) -> np.ndarray:
    """Sample raster values at grid cell centroids (fast alternative to polygonize+sjoin)."""
    centroids = grid_gdf.geometry.centroid
    if grid_gdf.crs is not None:
        centroids = gpd.GeoSeries(centroids, crs=grid_gdf.crs)

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        if grid_gdf.crs != raster_crs:
            centroids = centroids.to_crs(raster_crs)

        values = []
        for centroid in centroids.geometry:
            try:
                row, col = rasterio.transform.rowcol(src.transform, centroid.x, centroid.y)
                if 0 <= row < src.height and 0 <= col < src.width:
                    val = src.read(1)[row, col]
                    values.append(float(val))
                else:
                    values.append(np.nan)
            except (IndexError, Exception):
                values.append(np.nan)

    return np.array(values)


def load_raster_to_memory(raster_path: Path) -> dict[str, object]:
    """Load one raster band and metadata into memory for repeated sampling."""
    with rasterio.open(raster_path) as src:
        return {
            "array": src.read(1),
            "transform": src.transform,
            "nodata": src.nodata,
            "crs": src.crs,
            "height": src.height,
            "width": src.width,
            "path": raster_path,
        }


def sample_raster_memory_at_points(raster_mem: dict[str, object], points: gpd.GeoSeries) -> np.ndarray:
    """Sample in-memory raster values at point locations."""
    raster_crs = raster_mem["crs"]
    points_in_raster = points if points.crs == raster_crs else points.to_crs(raster_crs)

    arr = raster_mem["array"]
    transform = raster_mem["transform"]
    nodata = raster_mem["nodata"]
    height = int(raster_mem["height"])
    width = int(raster_mem["width"])

    xs = [p.x for p in points_in_raster.geometry]
    ys = [p.y for p in points_in_raster.geometry]
    rows, cols = rasterio.transform.rowcol(transform, xs, ys)

    sampled: list[float] = []
    for row, col in zip(rows, cols):
        if row < 0 or row >= height or col < 0 or col >= width:
            sampled.append(np.nan)
            continue

        val = float(arr[row, col])
        if nodata is None:
            sampled.append(val)
        elif isinstance(nodata, float) and np.isnan(nodata):
            sampled.append(np.nan if np.isnan(val) else val)
        else:
            sampled.append(np.nan if val == nodata else val)

    return np.array(sampled)


def create_uniform_master_grid(
    boundary_gdf: gpd.GeoDataFrame,
    target_crs: object,
    cell_size_m: float = 250,
    clip_to_boundary: bool = True,
) -> gpd.GeoDataFrame:
    """Create a uniform fishnet grid from boundary extent in target CRS."""
    boundary = boundary_gdf.to_crs(target_crs)
    minx, miny, maxx, maxy = boundary.total_bounds

    x_coords = np.arange(minx, maxx, cell_size_m)
    y_coords = np.arange(miny, maxy, cell_size_m)
    cells = [box(x, y, x + cell_size_m, y + cell_size_m) for x in x_coords for y in y_coords]

    grid = gpd.GeoDataFrame({"geometry": cells}, crs=target_crs)
    if clip_to_boundary:
        boundary_union = boundary.union_all()
        grid = grid[grid.geometry.intersects(boundary_union)].copy()

    return grid.reset_index(drop=True)


def qc_uniform_pixel_grid(master_grid: gpd.GeoDataFrame, value_col: str = "BUPR_value") -> None:
    """Quality-control check that master grid cells are uniform raster pixels."""
    if value_col not in master_grid.columns:
        raise ValueError(f"Missing expected column: {value_col}")

    areas = master_grid.geometry.area
    if areas.empty:
        raise ValueError("Master grid is empty.")

    min_area = float(areas.min())
    max_area = float(areas.max())
    if not np.isclose(min_area, max_area, rtol=1e-6, atol=1e-6):
        raise ValueError(f"Non-uniform grid detected. min_area={min_area}, max_area={max_area}")

    cell_size = np.sqrt(min_area)
    print(f"QC passed: uniform pixel grid (~{cell_size:.2f} m) with {len(master_grid)} cells")


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


def evaluate_led_vs_bupr(
    master_grid: gpd.GeoDataFrame,
    led_col: str = "LED_count",
    hisdac_col: str = "BUPR_value",
    state_code: str = "WA",
    analysis_year: str = "unknown",
) -> dict[str, float]:
    """Compare LED counts vs BUPR values with summary metrics and plots."""
    if led_col not in master_grid.columns:
        raise ValueError(f"Missing column: {led_col}")
    if hisdac_col not in master_grid.columns:
        raise ValueError(f"Missing column: {hisdac_col}")

    eval_df = master_grid[["HISDAC_id", led_col, hisdac_col]].copy()
    eval_df = eval_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[led_col, hisdac_col])

    if eval_df.empty:
        raise ValueError("No valid rows found for evaluation.")

    y_true = eval_df[hisdac_col].astype(float).to_numpy()
    y_pred = eval_df[led_col].astype(float).to_numpy()
    diff = y_pred - y_true

    rmse = float(np.sqrt(np.mean((diff) ** 2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(eval_df) > 1 else np.nan

    metrics = {
        "n_cells": float(len(eval_df)),
        "correlation": corr,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "mean_led": float(np.mean(y_pred)),
        "mean_bupr": float(np.mean(y_true)),
    }

    print("\nEvaluation Metrics (LED vs BUPR):")
    for k, v in metrics.items():
        if k == "n_cells":
            print(f"- {k}: {int(v)}")
        else:
            print(f"- {k}: {v:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_true, y_pred, s=8, alpha=0.35)
    min_val = float(min(np.nanmin(y_true), np.nanmin(y_pred)))
    max_val = float(max(np.nanmax(y_true), np.nanmax(y_pred)))
    axes[0].plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1)
    axes[0].set_xlabel("BUPR_value")
    axes[0].set_ylabel("LED_count")
    axes[0].set_title(f"{state_code} {analysis_year}: LED vs BUPR")

    axes[1].hist(diff, bins=40, color="steelblue", alpha=0.8)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Difference (LED_count - BUPR_value)")
    axes[1].set_ylabel("Grid cell count")
    axes[1].set_title("Distribution of Differences")

    plt.tight_layout()
    plt.show()

    return metrics
# %%
try:
    default_state_code = "WA"
    fbuy_fixed_file = HISDAC_DATASETS["FBUY"] / "FBUY.tif"

    # Load state boundary data
    state_boundaries = gpd.read_file(STATE_BOUNDARY_PATH)
    state_code = choose_state_code(state_boundaries, default_state=default_state_code)
    wa_boundary = state_boundaries[state_boundaries["STUSPS"] == state_code]

    if wa_boundary.empty:
        raise ValueError(f"No boundary geometry found for state code: {state_code}")

    print(f"Selected state: {state_code}")

    # Extract .tar files in each dataset folder (if any), then discover available years
    year_options = None
    for dataset_name, path in HISDAC_DATASETS.items():
        if dataset_name == "FBUY":
            continue
        extract_tar_files(path)
        dataset_years = get_available_years(path, dataset_name)
        if not dataset_years:
            continue
        year_options = dataset_years if year_options is None else sorted(set(year_options).intersection(dataset_years))

    if not year_options:
        raise ValueError("No common years found across datasets after extraction.")

    analysis_year = choose_year(year_options)
    print(f"\nSelected year: {analysis_year}")

    # 1) Load BUPR.
    bupr_path = HISDAC_DATASETS["BUPR"]
    bupr_file = find_dataset_file("BUPR", bupr_path, analysis_year)
    print(f"\n1) Found BUPR: {bupr_file}")

    if bupr_file.suffix.lower() != ".tif":
        raise ValueError("BUPR master-grid workflow expects a raster (.tif) file.")

    # 2) Create master grid from BUPR pixels (each pixel -> one square polygon).
    master_grid = raster_pixels_to_polygons(
        raster_path=bupr_file,
        clip_boundary=wa_boundary,
        value_col="BUPR_value",
    )
    print(f"2) Created raster-derived master grid with {len(master_grid)} cells")

    # 3) Create HISDAC_id.
    master_grid = master_grid.copy().reset_index(drop=True)
    master_grid["HISDAC_id"] = np.arange(1, len(master_grid) + 1)
    print(f"3) Added HISDAC_id to {len(master_grid)} grid cells")

    # QC: ensure this is a uniform pixel grid from source raster.
    qc_uniform_pixel_grid(master_grid, value_col="BUPR_value")

    # 4) Get centroids with HISDAC_id keys.
    centroid_table = gpd.GeoDataFrame(
        master_grid[["HISDAC_id"]].copy(),
        geometry=master_grid.geometry.centroid,
        crs=master_grid.crs,
    )
    print(f"4) Built centroid table with {len(centroid_table)} points")

    # 5) Load auxiliary rasters to memory.
    raster_memory: dict[str, dict[str, object]] = {}

    for name, path in HISDAC_DATASETS.items():
        if name == "BUPR":
            continue

        if name == "FBUY":
            data_file = fbuy_fixed_file
            if not data_file.exists():
                candidates = sorted(path.rglob("FBUY.tif"))
                if not candidates:
                    raise FileNotFoundError(f"FBUY file not found at {fbuy_fixed_file}")
                data_file = candidates[0]
        else:
            data_file = find_dataset_file(name, path, analysis_year)

        if data_file.suffix.lower() == ".tif":
            raster_memory[name] = load_raster_to_memory(data_file)
            print(f"5) Loaded {name} raster to memory: {data_file.name}")
        else:
            # Keep vector fallback for future mixed datasets.
            gdf = gpd.read_file(data_file)
            gdf = gpd.clip(gdf, wa_boundary.to_crs(gdf.crs))
            source_col = get_value_column(name, gdf)
            agg_func = AGG_RULES.get(name, "mean")
            out_col = f"{name}_{agg_func}"
            master_grid = aggregate_to_master_grid(master_grid, gdf, source_col, out_col, agg_func)
            print(f"Aggregated {name} -> {out_col}")

    # 6) Sample raster values at centroids.
    sampled_by_id = centroid_table[["HISDAC_id"]].copy()
    for name, raster_mem in raster_memory.items():
        agg_func = AGG_RULES.get(name, "mean")
        out_col = f"{name}_{agg_func}"
        sampled_values = sample_raster_memory_at_points(raster_mem, centroid_table.geometry)
        sampled_by_id[out_col] = sampled_values
        print(f"6) Sampled {name} at centroids -> {out_col}")

    # 7) Merge sampled raster attributes back to master grid by HISDAC_id.
    master_grid = master_grid.merge(sampled_by_id, on="HISDAC_id", how="left")
    print("7) Merged sampled raster values into master grid")

    # 8) Load LED data.
    led_file = TESTING_SET_PATH / f"{state_code}_LED_points-5070.gpkg"
    if not led_file.exists():
        raise FileNotFoundError(f"LED file not found for state {state_code}: {led_file}")
    led_results = gpd.read_file(led_file)

    # 9) Spatial join to add HISDAC_id to buildings.
    master_grid, led_with_id = attach_led_to_grid(master_grid, led_results)
    print("8) Linked LED points to master grid using HISDAC_id")

    # 10) Building count is merged in attach_led_to_grid as LED_count.
    print("9) Added LED_count to master grid")

    # Save outputs for downstream LED evaluation.
    master_out = TESTING_SET_PATH / f"{state_code}_{analysis_year}_HISDAC_master_grid.gpkg"
    led_out = TESTING_SET_PATH / f"{state_code}_{analysis_year}_LED_with_HISDAC_id.gpkg"
    master_grid.to_file(master_out, driver="GPKG")
    led_with_id.to_file(led_out, driver="GPKG")
    print(f"Saved master grid: {master_out}")
    print(f"Saved LED joined points: {led_out}")

    # Show quick summary of selected aggregated fields.
    summary_cols = [c for c in master_grid.columns if c.startswith(("BUPR_", "BUPL_", "BUA_", "BUI_", "FBUY_", "LED_count"))]
    print("\nMaster grid summary:")
    print(master_grid[["HISDAC_id"] + summary_cols].head())

    # Evaluate agreement between LED counts and BUPR values.
    metrics = evaluate_led_vs_bupr(
        master_grid=master_grid,
        led_col="LED_count",
        hisdac_col="BUPR_value",
        state_code=state_code,
        analysis_year=analysis_year,
    )

    metrics_out = TESTING_SET_PATH / f"{state_code}_{analysis_year}_LED_vs_BUPR_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_out, index=False)
    print(f"Saved evaluation metrics: {metrics_out}")

except Exception as e:
    print(f"An error occurred: {e}")

# %%
'''Evaluation is now performed automatically in the main workflow.'''
# %%
