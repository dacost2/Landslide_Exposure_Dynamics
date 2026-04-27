# %% [markdown]
# # Temporal Concordance Plotter (1:1 County & Statewide Validation)
# #### Daniel Acosta-Reyes
# %%
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from shapely.geometry import box
import sys
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# --- 1. User Inputs ---
# ==============================================================================
state_code = input("Enter 2-letter State Code (e.g., WA): ").strip().upper()
county_name = input(f"Enter County Name in {state_code} (e.g., 'King County', or leave blank for Statewide ONLY): ").strip().title()

is_county = len(county_name) > 0

# ==============================================================================
# --- 2. Setup Paths ---
# ==============================================================================
DATA_PATH = Path("/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics/Data")
PROD_PATH = DATA_PATH / "Production_set" / state_code
ANALYSIS_PATH = Path("/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics/Analysis")
OUTPUT_PATH = ANALYSIS_PATH / "temporal_concordance_plots"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# File Paths
det_centroids_path = PROD_PATH / f"{state_code}_Deterministic_Pixel_Centroids.gpkg"
prob_points_path = PROD_PATH / f"{state_code}_Probabilistic_Building_Points.gpkg"
led_engine_path = PROD_PATH / f"{state_code}_LED_Monte_Carlo_Engine.parquet"

# NHGIS Paths
NHGIS_40_PATH = DATA_PATH / 'NHGIS/nhgis0004_COUNTY_csv/nhgis0004_ds78_1940_county.csv'
NHGIS_TS_PATH = DATA_PATH / 'NHGIS/nhgis0004_COUNTY_csv/nhgis0004_ts_nominal_county.csv'
SHAPE_40_PATH = DATA_PATH / 'NHGIS/nhgis0005_COUNTY_shape/nhgis0005_shapefile_tl2000_us_county_1940.zip'
SHAPE_2020_PATH = DATA_PATH / 'NHGIS/nhgis0005_COUNTY_shape/nhgis0005_shapefile_tl2020_us_county_2020.zip'

def normalize_gisjoin(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()

# ==============================================================================
# --- 3. Load Statewide Data Once (Memory Optimization) ---
# ==============================================================================
print("\n1) Loading Master Datasets into Memory...")

# Load Deterministic
state_det = gpd.read_file(det_centroids_path)
data_crs = state_det.crs

# Load Probabilistic Spatial & Engine
state_prob_spatial = gpd.read_file(prob_points_path)
led_engine = pd.read_parquet(led_engine_path)

# Merge Probabilistic Arrays
state_prob = state_prob_spatial.merge(led_engine[['prob_distribution']], left_index=True, right_index=True, how='inner')

# Pre-load NHGIS Geometries and Data
print("   -> Loading NHGIS Census Files...")
shape_40 = gpd.read_file(f"zip://{SHAPE_40_PATH}").to_crs(data_crs)
shape_data = gpd.read_file(f"zip://{SHAPE_2020_PATH}").to_crs(data_crs)
nhgis_40 = pd.read_csv(NHGIS_40_PATH, encoding="iso-8859-1")
nhgis_data = pd.read_csv(NHGIS_TS_PATH, encoding="iso-8859-1")

for df in [nhgis_data, nhgis_40, shape_40, shape_data]:
    if "NHGISCODE" in df.columns and "GISJOIN" not in df.columns:
        df.rename(columns={"NHGISCODE": "GISJOIN"}, inplace=True)
    df["GISJOIN"] = normalize_gisjoin(df["GISJOIN"])

# ==============================================================================
# --- 4. Locate Target County Geometry ---
# ==============================================================================
county_extent = None
if is_county:
    print(f"\n2) Locating {county_name} in NHGIS Spatial Data...")
    
    # Create a bounding box of the loaded state data to instantly filter U.S. counties
    state_bounds = box(*state_det.total_bounds)
    state_bounds_gdf = gpd.GeoDataFrame(geometry=[state_bounds], crs=data_crs)
    
    # Find all NHGIS counties that intersect this state's bounding box
    possible_counties = gpd.sjoin(shape_data, state_bounds_gdf, how='inner', predicate='intersects').drop(columns=['index_right'], errors='ignore')
    
    # Exact text match on the county name
    county_match = possible_counties[possible_counties['NAMELSAD'].str.lower() == county_name.lower()]
    
    if county_match.empty:
        print(f"[ERROR] Could not find '{county_name}' within the spatial bounds of {state_code}.")
        print(f"Available counties in this area include: {', '.join(possible_counties['NAMELSAD'].head().tolist())}...")
        sys.exit(1)
        
    county_extent = county_match.iloc[[0]] # Extract the exact geometry polygon
    print(f"   -> Successfully isolated geometry for {county_name}.")

# ==============================================================================
# --- CORE ENGINE: Calculate and Plot Function ---
# ==============================================================================
def process_and_plot_region(det_gdf, prob_gdf, region_name, is_local_flag, extent_geom=None):
    print(f"\n--- Processing Temporal Concordance for: {region_name} ---")
    
    DECADES = np.arange(1920, 2025, 10)
    YEARS_5 = np.arange(1920, 2025, 5)

    # 1. Prep Deterministic Baseline
    hazard_cols = ['high', 'moderate', 'low', 'none']
    for c in hazard_cols:
        if c not in det_gdf.columns: det_gdf[c] = 0
    det_gdf['Total_LED_2020'] = det_gdf[hazard_cols].fillna(0).sum(axis=1)
    det_gdf['Total_HISDAC_BUPL'] = det_gdf['C_BUPL2020'].fillna(0)

    # 2. Prep Probabilistic Baseline
    if not prob_gdf.empty:
        prob_matrix = np.stack(prob_gdf['prob_distribution'].values)
    else:
        prob_matrix = np.array([])

    plot_data = []

    # 3. Calculate Model Volumes
    print("   -> Calculating Decadal Model Volumes...")
    for dec in DECADES:
        record = {"Year": dec}
        record["HISDAC_BUPL"] = det_gdf[f"C_BUPL{dec}"].fillna(0).sum()
        
        if f"C_BUPR{dec}" in det_gdf.columns:
            record["HISDAC_BUPR"] = det_gdf[f"C_BUPR{dec}"].fillna(0).sum()
        else:
            record["HISDAC_BUPR"] = np.nan
            
        cum_hisdac = det_gdf[f'C_BUPL{dec}'].fillna(0)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(det_gdf['Total_HISDAC_BUPL'] > 0, cum_hisdac / det_gdf['Total_HISDAC_BUPL'], 0)
        if dec == 2020:
            ratio = np.where(det_gdf['Total_HISDAC_BUPL'] == 0, 1.0, ratio)
        
        record["Deterministic"] = (det_gdf['Total_LED_2020'] * ratio).sum()
        
        if prob_matrix.size > 0:
            idx = np.where(YEARS_5 == dec)[0][0]
            record["Probabilistic"] = np.sum(prob_matrix[:, :idx+1])
        else:
            record["Probabilistic"] = 0
            
        plot_data.append(record)

    plot_df = pd.DataFrame(plot_data)

    # 4. Extract NHGIS Census Data
# 4. Extract NHGIS Census Data
    print("   -> Extracting NHGIS Spatial Boundaries...")
    if is_local_flag and extent_geom is not None:
        # THE FIX: Only pass [['geometry']] so we don't bring in conflicting columns!
        valid_40 = gpd.sjoin(shape_40, extent_geom[['geometry']], how="inner", predicate="intersects")["GISJOIN"].unique()
        valid_2020 = gpd.sjoin(shape_data, extent_geom[['geometry']], how="inner", predicate="intersects")["GISJOIN"].unique()
    else:
        valid_40 = gpd.sjoin(shape_40, det_gdf[['geometry']], how="inner", predicate="intersects")["GISJOIN"].unique()
        valid_2020 = gpd.sjoin(shape_data, det_gdf[['geometry']], how="inner", predicate="intersects")["GISJOIN"].unique()

    local_nhgis_40 = nhgis_40[nhgis_40["GISJOIN"].isin(valid_40)]
    local_nhgis_data = nhgis_data[nhgis_data["GISJOIN"].isin(valid_2020)]

    decade_sources = {
        1940: (local_nhgis_40, "BXR001"),
        1970: (local_nhgis_data, "A41AA1970"),
        1980: (local_nhgis_data, "A41AA1980"),
        1990: (local_nhgis_data, "A41AA1990"),
        2000: (local_nhgis_data, "A41AA2000"),
        2010: (local_nhgis_data, "A41AA2010"),
        2020: (local_nhgis_data, "A41AA2020"),
    }

    nhgis_records = []
    for dec, (src_df, col) in decade_sources.items():
        if col in src_df.columns:
            val = pd.to_numeric(src_df[col], errors="coerce").sum()
            nhgis_records.append({"Year": dec, "NHGIS_Units": val})

    nhgis_df = pd.DataFrame(nhgis_records)
    plot_df = plot_df.merge(nhgis_df, on="Year", how="left")

    # 5. Rendering Plot
    print("   -> Rendering and Saving Plot...")
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(plot_df["Year"], plot_df["HISDAC_BUPR"], marker="x", lw=2, linestyle=':', color="#8e44ad", label="HISDAC BUPR (Historical Res. Units)")
    ax.plot(plot_df["Year"], plot_df["HISDAC_BUPL"], marker="v", lw=2, linestyle='-.', color="#3498db", label="HISDAC BUPL (Historical Footprints)")
    ax.plot(plot_df["Year"], plot_df["Deterministic"], marker="^", lw=2.5, linestyle='--', color="#e67e22", label="Deterministic Baseline (Distributed LED)")
    ax.plot(plot_df["Year"], plot_df["Probabilistic"], marker="s", lw=2.5, linestyle='-', color="#27ae60", label="Probabilistic Baseline (Bayesian LED)")

    valid_nhgis = plot_df.dropna(subset=['NHGIS_Units'])
    ax.plot(valid_nhgis["Year"], valid_nhgis["NHGIS_Units"], marker="o", markersize=8, lw=3, color="#2c3e50", label="NHGIS (Census Housing Units)")

    ax.set_title(f"Temporal Concordance: {region_name}\n1920–2020 Building and Housing Inventories", fontsize=18, fontweight='bold')
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Cumulative Volume (Count)", fontsize=14)
    ax.set_xticks(DECADES)
    ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)
    ax.axvspan(1940, 1970, color='gray', alpha=0.1, label='NHGIS Data Gap')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

    plt.tight_layout()

    filename = f"{state_code}_{region_name.split(',')[0].replace(' ', '')}_Temporal_Concordance.png" if is_local_flag else f"{state_code}_Statewide_Temporal_Concordance.png"
    fig.savefig(OUTPUT_PATH / filename, dpi=300)
    plt.close(fig)

    print(f"   [DONE] Plot saved to: {filename}")

# ==============================================================================
# --- EXECUTION PIPELINE ---
# ==============================================================================

# 1. ALWAYS plot the full State first
if not is_county:
    process_and_plot_region(state_det.copy(), state_prob.copy(), f"State of {state_code}", is_local_flag=False)
else:
    # We still run the state to have it, then run the county
    process_and_plot_region(state_det.copy(), state_prob.copy(), f"State of {state_code}", is_local_flag=False)

    print(f"\n3) Clipping highly detailed datasets precisely to {county_name} Extent...")
    # Clip pixels and points strictly to the exact County Polygon
    county_det = gpd.clip(state_det, county_extent)
    county_prob_clipped = gpd.clip(state_prob, county_extent)

    process_and_plot_region(county_det, county_prob_clipped, f"{county_name}, {state_code}", is_local_flag=True, extent_geom=county_extent)

print("\n=== ALL TEMPORAL CONCORDANCE PLOTS COMPLETED SUCCESSFULLY ===")