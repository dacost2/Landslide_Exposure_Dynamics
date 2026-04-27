# %% [markdown]
# # Phase 4: Macro-Validation (NHGIS Referee - 1940 Baseline)
# #### Daniel Acosta-Reyes
# %%
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# --- 1. Argparse Setup for Master Controller ---
parser = argparse.ArgumentParser(description="Run NHGIS Validation for a specific state.")
parser.add_argument("--state", type=str, required=False, help="2-letter state code (e.g., WA)")
args, unknown = parser.parse_known_args()
if args.state:
    STATE_CODE = args.state.upper()
elif "ipykernel" in sys.argv[0] or "ipykernel" in sys.modules:
    STATE_CODE = os.getenv("STATE_CODE", "AL").upper()
    print(f"[INFO] No --state provided in interactive session. Using STATE_CODE={STATE_CODE}.")
else:
    parser.error("the following arguments are required: --state")

print(f"=== Starting NHGIS Validation Engine for {STATE_CODE} ===")

# --- 2. Paths Setup ---
DATA_PATH = Path("/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics/Data")
PRODUCTION_SET_PATH = DATA_PATH / "Production_set" / STATE_CODE
ANALYSIS_PATH = Path("/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics/Analysis")
PRODUCTION_ANALYTICS = ANALYSIS_PATH / "Production_analytics" / STATE_CODE

PRODUCTION_ANALYTICS.mkdir(parents=True, exist_ok=True)

# NHGIS Paths
NHGIS_40_PATH = DATA_PATH / 'NHGIS/nhgis0004_COUNTY_csv/nhgis0004_ds78_1940_county.csv'
NHGIS_TS_PATH = DATA_PATH / 'NHGIS/nhgis0004_COUNTY_csv/nhgis0004_ts_nominal_county.csv'
SHAPE_40_PATH = DATA_PATH / 'NHGIS/nhgis0005_COUNTY_shape/nhgis0005_shapefile_tl2000_us_county_1940.zip'
SHAPE_2020_PATH = DATA_PATH / 'NHGIS/nhgis0005_COUNTY_shape/nhgis0005_shapefile_tl2020_us_county_2020.zip'

def normalize_gisjoin(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()

# UPDATE 1: Standardize the 1940 Timeline
YEARS = np.arange(1940, 2025, 5)

# --- 3. Load Data ---
print("1) Loading Datasets...")
led_spatial = gpd.read_file(PRODUCTION_SET_PATH / f"{STATE_CODE}_Probabilistic_Building_Points.gpkg", columns=['HISDAC_id', 'geometry'])
led_engine = pd.read_parquet(PRODUCTION_SET_PATH / f"{STATE_CODE}_LED_Monte_Carlo_Engine.parquet")
det_centroids = gpd.read_file(PRODUCTION_SET_PATH / f"{STATE_CODE}_Deterministic_Pixel_Centroids.gpkg")

# Merge probability arrays to geometry
led_inventory = led_spatial.merge(led_engine[['prob_distribution']], left_index=True, right_index=True, how='inner')

shape_40 = gpd.read_file(f"zip://{SHAPE_40_PATH}")
shape_data = gpd.read_file(f"zip://{SHAPE_2020_PATH}")
nhgis_40 = pd.read_csv(NHGIS_40_PATH, encoding="iso-8859-1")
nhgis_data = pd.read_csv(NHGIS_TS_PATH, encoding="iso-8859-1")

for df in [nhgis_data, nhgis_40]:
    if "NHGISCODE" in df.columns and "GISJOIN" not in df.columns:
        df.rename(columns={"NHGISCODE": "GISJOIN"}, inplace=True)
    df["GISJOIN"] = normalize_gisjoin(df["GISJOIN"])

shape_40["GISJOIN"] = normalize_gisjoin(shape_40["GISJOIN"])
shape_data["GISJOIN"] = normalize_gisjoin(shape_data["GISJOIN"])

# --- 4. Spatial Joins ---
print("2) Running Spatial Joins to Historical Counties...")
target_crs = led_inventory.crs
shape_40 = shape_40.to_crs(target_crs)
shape_data = shape_data.to_crs(target_crs)

shape_40_join = shape_40[["GISJOIN", "geometry"]].rename(columns={"GISJOIN": "GISJOIN_40"})
shape_data_join = shape_data[["GISJOIN", "geometry"]].rename(columns={"GISJOIN": "GISJOIN_2020"})

led_joined = gpd.sjoin(led_inventory, shape_40_join, how="inner", predicate="intersects").drop(columns=["index_right"], errors="ignore")
led_joined = gpd.sjoin(led_joined, shape_data_join, how="inner", predicate="intersects").drop(columns=["index_right"], errors="ignore")

det_joined = gpd.sjoin(det_centroids, shape_40_join, how="inner", predicate="intersects").drop(columns=["index_right"], errors="ignore")
det_joined = gpd.sjoin(det_joined, shape_data_join, how="inner", predicate="intersects").drop(columns=["index_right"], errors="ignore")

# --- 5. Calculate Cumulative Expected Values ---
print("3) Calculating Cumulative Decadal Values...")

# A) Probabilistic (Using the Bayesian Arrays)
prob_matrix = np.stack(led_joined['prob_distribution'].values)
target_decades = [1940, 1970, 1980, 1990, 2000, 2010, 2020]

# UPDATE 2: Dynamic slice indexing based on the YEARS array
def get_slice_idx(year): 
    return np.where(YEARS == year)[0][0] + 1

for decade in target_decades:
    idx = get_slice_idx(decade)
    led_joined[f'prob_cum_{decade}'] = np.sum(prob_matrix[:, :idx], axis=1)

# B) Deterministic (Area-Level Method)
hazard_cols = ['high', 'moderate', 'low', 'none']
for c in hazard_cols:
    if c not in det_joined.columns: det_joined[c] = 0
det_joined['Total_LED_2020'] = det_joined[hazard_cols].sum(axis=1)
det_joined['Total_HISDAC'] = det_joined['C_BUPL2020'].fillna(0)

# UPDATE 3: Implement Tier 2 Fallback logic for the county validation
for decade in target_decades:
    cum_hisdac = det_joined[f'C_BUPL{decade}'].fillna(0).values
    
    # Dynamically accumulate the LED fallback proxy up to this specific decade
    cum_led_fallback = np.zeros(len(det_joined))
    for y in YEARS[YEARS <= decade]:
        if f'LED_{y}' in det_joined.columns:
            cum_led_fallback += det_joined[f'LED_{y}'].fillna(0).values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_hisdac = np.where(det_joined['Total_HISDAC'] > 0, cum_hisdac / det_joined['Total_HISDAC'], 0)
        ratio_led = np.where(det_joined['Total_LED_2020'] > 0, cum_led_fallback / det_joined['Total_LED_2020'], 0)
        
    # Apply Tier logic
    ratio = np.where(det_joined['Total_HISDAC'] > 0, ratio_hisdac, ratio_led)
        
    if decade == 2020:
        ratio = 1.0
        
    det_joined[f'det_cum_{decade}'] = det_joined['Total_LED_2020'] * ratio

# --- 6. Aggregate & Merge with NHGIS ---
print("4) Aggregating to County Level...")
valid_gisjoins_40 = led_joined["GISJOIN_40"].unique()
valid_gisjoins_2020 = led_joined["GISJOIN_2020"].unique()

nhgis_40 = nhgis_40[nhgis_40["GISJOIN"].isin(valid_gisjoins_40)].copy()
nhgis_data = nhgis_data[nhgis_data["GISJOIN"].isin(valid_gisjoins_2020)].copy()

# 1940 Aggregation
county_1940 = pd.DataFrame({'GISJOIN': valid_gisjoins_40})
p_40 = led_joined.groupby("GISJOIN_40", as_index=False)['prob_cum_1940'].sum().rename(columns={"GISJOIN_40": "GISJOIN"})
d_40 = det_joined.groupby("GISJOIN_40", as_index=False)['det_cum_1940'].sum().rename(columns={"GISJOIN_40": "GISJOIN"})

county_1940 = county_1940.merge(p_40, on='GISJOIN', how='left').merge(d_40, on='GISJOIN', how='left').fillna(0)
nhgis_40 = nhgis_40.merge(county_1940, on="GISJOIN", how="left").fillna(0)

# Modern Aggregation
mod_decades = [1970, 1980, 1990, 2000, 2010, 2020]
p_agg = {f'prob_cum_{d}': 'sum' for d in mod_decades}
d_agg = {f'det_cum_{d}': 'sum' for d in mod_decades}

p_mod = led_joined.groupby("GISJOIN_2020", as_index=False).agg(p_agg).rename(columns={"GISJOIN_2020": "GISJOIN"})
d_mod = det_joined.groupby("GISJOIN_2020", as_index=False).agg(d_agg).rename(columns={"GISJOIN_2020": "GISJOIN"})

county_mod = pd.DataFrame({'GISJOIN': valid_gisjoins_2020}).merge(p_mod, on='GISJOIN', how='left').merge(d_mod, on='GISJOIN', how='left').fillna(0)
nhgis_data = nhgis_data.merge(county_mod, on="GISJOIN", how="left").fillna(0)

# --- 7. Validation Analytics ---
print("5) Computing Validation Metrics...")
decade_sources = {
    1940: (nhgis_40, "BXR001"),
    1970: (nhgis_data, "A41AA1970"),
    1980: (nhgis_data, "A41AA1980"),
    1990: (nhgis_data, "A41AA1990"),
    2000: (nhgis_data, "A41AA2000"),
    2010: (nhgis_data, "A41AA2010"),
    2020: (nhgis_data, "A41AA2020"),
}

metrics_rows = []
state_plot_data = []

def resolve_col(df, col): return col if col in df.columns else None

for decade, (src_df, nhgis_col) in decade_sources.items():
    nhgis_c = resolve_col(src_df, nhgis_col)
    if not nhgis_c: continue

    plot_df = src_df[["GISJOIN", nhgis_c, f'prob_cum_{decade}', f'det_cum_{decade}']].copy()
    plot_df[nhgis_c] = pd.to_numeric(plot_df[nhgis_c], errors="coerce")
    plot_df = plot_df.dropna()

    y_true = plot_df[nhgis_c].values
    y_prob = plot_df[f'prob_cum_{decade}'].values
    y_det = plot_df[f'det_cum_{decade}'].values
    
    state_plot_data.append({
        "decade": decade, "total_nhgis": y_true.sum(), 
        "expected_prob": y_prob.sum(), "expected_det": y_det.sum()
    })

    if len(plot_df) > 1:
        metrics_rows.append({
            "Decade": decade, "Counties": len(plot_df),
            "Prob_Mean_Diff": np.mean(y_prob - y_true), "Det_Mean_Diff": np.mean(y_det - y_true),
            "Prob_Corr_r": np.corrcoef(y_true, y_prob)[0, 1], "Det_Corr_r": np.corrcoef(y_true, y_det)[0, 1],
            "Prob_RMSE": np.sqrt(mean_squared_error(y_true, y_prob)), "Det_RMSE": np.sqrt(mean_squared_error(y_true, y_det)),
            "Prob_MAE": mean_absolute_error(y_true, y_prob), "Det_MAE": mean_absolute_error(y_true, y_det)
        })

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(PRODUCTION_ANALYTICS / f"{STATE_CODE}_NHGIS_Validation_Metrics.csv", index=False)

# --- 8. Plotting ---
print("6) Generating Dual Concordance Plot (Absolute vs Trajectory)...")
state_plot_df = pd.DataFrame(state_plot_data).sort_values("decade")

# Calculate Normalized Trajectories (0 to 100%)
state_plot_df['norm_nhgis'] = (state_plot_df['total_nhgis'] / state_plot_df['total_nhgis'].max()) * 100
state_plot_df['norm_prob'] = (state_plot_df['expected_prob'] / state_plot_df['expected_prob'].max()) * 100
state_plot_df['norm_det'] = (state_plot_df['expected_det'] / state_plot_df['expected_det'].max()) * 100

# UPDATE 4: Dual Subplots to handle the "Housing Units vs Building Footprints" argument
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(f"Statewide Macro-Validation: {STATE_CODE}\nHousing Units (NHGIS) vs. Built Environment (HSA Framework)", fontsize=18, fontweight='bold', y=1.05)

# Plot 1: Absolute Volumes
ax1.plot(state_plot_df["decade"], state_plot_df["total_nhgis"], marker="o", lw=3, label="NHGIS (Housing Units)", color="#2c3e50")
ax1.plot(state_plot_df["decade"], state_plot_df["expected_prob"], marker="s", lw=2, label="Probabilistic (LED Footprints)", color="#27ae60")
ax1.plot(state_plot_df["decade"], state_plot_df["expected_det"], marker="^", lw=2, linestyle='--', label="Deterministic (HSA Baseline)", color="#e67e22")
ax1.fill_between(state_plot_df["decade"], state_plot_df["total_nhgis"], state_plot_df["expected_prob"], color='gray', alpha=0.15, label='Multi-Unit Density Gap')

ax1.set_title("Absolute Cumulative Volumes", fontsize=14, fontweight='bold')
ax1.set_xlabel("Decade", fontsize=12)
ax1.set_ylabel("Total Count", fontsize=12)
ax1.set_xticks(state_plot_df["decade"])
ax1.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)
ax1.legend(loc='upper left', fontsize=10)

# Plot 2: Normalized Sprawl Trajectory
ax2.plot(state_plot_df["decade"], state_plot_df["norm_nhgis"], marker="o", lw=3, label="NHGIS Sprawl Trajectory", color="#2c3e50")
ax2.plot(state_plot_df["decade"], state_plot_df["norm_prob"], marker="s", lw=2, label="Probabilistic Sprawl Trajectory", color="#27ae60")
ax2.plot(state_plot_df["decade"], state_plot_df["norm_det"], marker="^", lw=2, linestyle='--', label="Deterministic Sprawl Trajectory", color="#e67e22")

ax2.set_title("Normalized Exposure Trajectory (0-100% Buildout)", fontsize=14, fontweight='bold')
ax2.set_xlabel("Decade", fontsize=12)
ax2.set_ylabel("Percentage of 2020 Built Environment (%)", fontsize=12)
ax2.set_xticks(state_plot_df["decade"])
ax2.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)
ax2.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig(PRODUCTION_ANALYTICS / f"{STATE_CODE}_NHGIS_Dual_Concordance.png", dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"\n=== SCRIPT 4 COMPLETE FOR {STATE_CODE} ===")