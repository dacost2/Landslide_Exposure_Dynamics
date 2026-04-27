# %% [markdown]
# # Deterministic vs Probabilistic Exposure Trajectories (Production - 1940 Baseline)
# #### Daniel Acosta-Reyes
# %%
import pandas as pd
import numpy as np
from scipy.stats import norm
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import geopandas as gpd
import argparse
import os
import sys

# --- 1. Argparse Setup for Master Controller ---
parser = argparse.ArgumentParser(description="Run Methods and Analytics for a specific state.")
parser.add_argument("--state", type=str, required=False, help="2-letter state code (e.g., WA)")
args, unknown = parser.parse_known_args()
if args.state:
    STATE_CODE = args.state.upper()
elif "ipykernel" in sys.argv[0] or "ipykernel" in sys.modules:
    STATE_CODE = os.getenv("STATE_CODE", "AL").upper()
    print(f"[INFO] No --state provided in interactive session. Using STATE_CODE={STATE_CODE}.")
else:
    parser.error("the following arguments are required: --state")

# --- 2. Setup & Load Datasets ---
DATA_PATH = Path("/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics/Data")
ANALYSIS_PATH = Path("/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics/Analysis")

PRODUCTION_SET_PATH = DATA_PATH / "Production_set" / STATE_CODE
PRODUCTION_ANALYTICS = ANALYSIS_PATH / "Production_analytics" / STATE_CODE

# Ensure analytics directory exists for this specific state
PRODUCTION_ANALYTICS.mkdir(parents=True, exist_ok=True)

# UPDATE 1: Shift the baseline year to 1940
YEARS = np.arange(1940, 2025, 5)
ARRAY_LEN = len(YEARS) # Will be 17

print(f"=== Starting Methods Pipeline for {STATE_CODE} ===")
print("1) Loading Datasets...")

# Load tabular data
master_df = pd.read_parquet(PRODUCTION_SET_PATH / f"{STATE_CODE}_Master_Spatiotemporal_Matrix.parquet")
led_df = pd.read_parquet(PRODUCTION_SET_PATH / f"{STATE_CODE}_LED_Joined_Buildings.parquet")

# Load spatial data for final exports
led_gdf = gpd.read_file(PRODUCTION_SET_PATH / f"{STATE_CODE}_LED_Joined_Buildings.gpkg")
spatial_anchor = gpd.read_file(PRODUCTION_SET_PATH / f"{STATE_CODE}_HISDAC_Spatial_Anchor.gpkg")

# Ensure susc_class is clean
if 'susc_class' not in led_df.columns:
    raise KeyError("Column 'susc_class' is missing from led_df.")
led_df['susc_class'] = led_df['susc_class'].astype(str).str.lower().str.strip()
hazard_classes = ['none', 'low', 'moderate', 'high']

master_idx = master_df.set_index('HISDAC_id')

# Dictionaries to store the final exposure trajectories
cumulative_exposure_P = {c: np.zeros(ARRAY_LEN) for c in hazard_classes}
cumulative_exposure_D = {c: np.zeros(ARRAY_LEN) for c in hazard_classes}

# ==============================================================================
# --- 3. Probabilistic Execution (Tier 3 Core) ---
# ==============================================================================
print("\n2a) Running Probabilistic Engine...")

def generate_probability_array(row):
    hid = row['HISDAC_id']
    if hid not in master_idx.index:
        # UPDATE 2: Dynamic array length based on the YEARS variable
        prob_array = np.zeros(ARRAY_LEN)
        year_idx = np.where(YEARS == row['semi_decade'])[0][0]
        prob_array[year_idx] = 1.0
        return prob_array

    pixel = master_idx.loc[hid]
    led_year = row['semi_decade']
    source_nsi = row.get('source_nsi', 'unknown')
    stories = row.get('num_story', 1)
    if pd.isna(stories): stories = 1
    
    hisdac_bupl = np.array([pixel[f'D_BUPL{y}'] for y in YEARS])
    led_counts = np.array([pixel[f'LED_{y}'] for y in YEARS])
    density = pixel['DENSITY']
    fbuy = pixel['FBUY']
    
    prob_array = np.zeros(ARRAY_LEN)
    
    if pd.isna(fbuy) or np.sum(hisdac_bupl) == 0:
        year_idx = np.where(YEARS == led_year)[0][0]
        prob_array[year_idx] = 1.0
        return prob_array

    if density > 3.0:
        if stories > 3:
            hisdac_bupr = np.array([pixel[f'D_BUPR{y}'] for y in YEARS])
            if np.sum(hisdac_bupr) > 0: return hisdac_bupr / np.sum(hisdac_bupr)

    if source_nsi == 'nsi_estimated':
        # UPDATE 3: Dynamic slice for the last 20 years (last 5 elements)
        recent_idx = slice(ARRAY_LEN - 5, ARRAY_LEN)
        deficits = np.maximum(0, hisdac_bupl[recent_idx] - led_counts[recent_idx])
        if np.sum(deficits) > 0:
            prob_array[recent_idx] = deficits / np.sum(deficits)
            return prob_array

    # 1. Calculate the raw historical distribution
    base_dist = hisdac_bupl / np.sum(hisdac_bupl)
    
    # 2. Widen the Gaussian to account for "Effective Year Built" renovations in NSI
    gaussian_weights = norm.pdf(YEARS, loc=led_year, scale=15) 
    prob_array = base_dist * gaussian_weights
    
    # 3. If there is overlap, return the smoothed Bayesian distribution
    if np.sum(prob_array) > 0: 
        return prob_array / np.sum(prob_array)
        
    # 4. If NSI completely misses the HISDAC history, TRUST HISDAC.
    return base_dist

tqdm.pandas(desc="Calculating Probabilities")
led_df['prob_distribution'] = led_df.progress_apply(generate_probability_array, axis=1)

led_df['map_year_built'] = led_df['prob_distribution'].apply(lambda x: YEARS[np.argmax(x)])

prob_matrix = np.stack(led_df['prob_distribution'].values)
cum_prob_matrix = np.cumsum(prob_matrix, axis=1)

for c in hazard_classes:
    mask = led_df['susc_class'] == c
    if mask.sum() > 0:
        cumulative_exposure_P[c] = np.sum(cum_prob_matrix[mask], axis=0)

# ==============================================================================
# --- 4. Deterministic Execution (Area-Level Method) ---
# ==============================================================================
print("\n2b) Running Deterministic Engine (Area-Level Method)...")

led_exposed_counts = led_df.groupby(['HISDAC_id', 'susc_class']).size().unstack(fill_value=0)
det_matrix = master_df.join(led_exposed_counts, on='HISDAC_id', how='left')

for c in hazard_classes:
    if c not in det_matrix.columns:
        det_matrix[c] = 0
det_matrix[hazard_classes] = det_matrix[hazard_classes].fillna(0)

det_matrix['Total_LED_2020'] = det_matrix[hazard_classes].sum(axis=1)
det_matrix['Total_HISDAC'] = det_matrix['C_BUPL2020'].fillna(0)
det_matrix['Cumulative_LED_Fallback'] = 0

# UPDATE 4: Implement the HSF Tiers of Confidence Tagging (Pixel Level)
det_matrix['Confidence_Tier'] = np.where(
    det_matrix['Total_HISDAC'] > 0, 
    'Tier 1 (High - HISDAC Concordance)', 
    'Tier 2 (Moderate - LED Fallback)'
)

for y_idx, y in enumerate(YEARS):
    cumulative_hisdac = det_matrix[f'C_BUPL{y}'].fillna(0).values
    
    if f'LED_{y}' in det_matrix.columns:
        det_matrix['Cumulative_LED_Fallback'] += det_matrix[f'LED_{y}'].fillna(0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_hisdac = np.where(det_matrix['Total_HISDAC'] > 0,
                                cumulative_hisdac / det_matrix['Total_HISDAC'],
                                0)
        
        ratio_led = np.where(det_matrix['Total_LED_2020'] > 0,
                             det_matrix['Cumulative_LED_Fallback'] / det_matrix['Total_LED_2020'],
                             0)
                             
    growth_ratio = np.where(det_matrix['Total_HISDAC'] > 0, ratio_hisdac, ratio_led)
    
    if y == 2020:
        growth_ratio = 1.0
        
    for c in hazard_classes:
        exposed_in_pixel = det_matrix[c].values * growth_ratio
        cumulative_exposure_D[c][y_idx] = exposed_in_pixel.sum()

# ==============================================================================
# --- 5. Exposure Rate Analytics & Visualization ---
# ==============================================================================
print("\n3) Generating Analytics and Plots...")

total_built_P = np.zeros(ARRAY_LEN)
total_built_D = np.zeros(ARRAY_LEN)

for c in hazard_classes:
    total_built_P += cumulative_exposure_P[c]
    total_built_D += cumulative_exposure_D[c]

rate_P = {c: np.zeros(ARRAY_LEN) for c in hazard_classes}
rate_D = {c: np.zeros(ARRAY_LEN) for c in hazard_classes}

for c in hazard_classes:
    with np.errstate(divide='ignore', invalid='ignore'):
        rate_P[c] = np.where(total_built_P > 0, (cumulative_exposure_P[c] / total_built_P) * 100, 0)
        rate_D[c] = np.where(total_built_D > 0, (cumulative_exposure_D[c] / total_built_D) * 100, 0)

color_map = {'high': '#d62728', 'moderate': '#ff7f0e', 'low': '#f1c40f', 'none': '#7f8c8d'}

# --- PLOT 1: Cumulative Exposure ---
fig1, ax1 = plt.subplots(figsize=(14, 8))
fig1.suptitle(f"Cumulative Landslide Exposure: Probabilistic vs Deterministic ({STATE_CODE})", fontsize=16, fontweight='bold')

for c in ['high', 'moderate', 'low']:
    color = color_map[c]
    ax1.plot(YEARS, cumulative_exposure_P[c], color=color, linestyle='-', linewidth=3, marker='o', label=f"{c.capitalize()} (Probabilistic)")
    ax1.plot(YEARS, cumulative_exposure_D[c], color=color, linestyle='--', linewidth=2, alpha=0.7, label=f"{c.capitalize()} (Deterministic Baseline)")

ax1.set_title("State-level Methodological Comparison", fontsize=12)
ax1.set_ylabel("Total Cumulative Buildings Exposed")
ax1.set_xlabel("Year")
ax1.set_xticks(YEARS)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='upper left', ncol=2)

fig1.tight_layout()
fig1.savefig(PRODUCTION_ANALYTICS / f"{STATE_CODE}_Exposure_Cumulative_Comparison.png", dpi=300, bbox_inches='tight')
plt.close(fig1)

# --- PLOT 2: Exposure Rates ---
fig2, ax2 = plt.subplots(figsize=(14, 8))
fig2.suptitle(f"Landslide Exposure Rate E(t): Probabilistic vs Deterministic ({STATE_CODE})", fontsize=16, fontweight='bold')

for c in ['high', 'moderate', 'low']:
    color = color_map[c]
    ax2.plot(YEARS, rate_P[c], color=color, linestyle='-', linewidth=3, marker='o', label=f"{c.capitalize()} Rate (Probabilistic)")
    ax2.plot(YEARS, rate_D[c], color=color, linestyle='--', linewidth=2, alpha=0.7, label=f"{c.capitalize()} Rate (Deterministic)")

ax2.set_title("E(t) = Exposed Buildings / Total Buildings", fontsize=12)
ax2.set_ylabel("Exposure Rate (%)")
ax2.set_xlabel("Year")
ax2.set_xticks(YEARS)
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='upper left', ncol=2)

fig2.tight_layout()
fig2.savefig(PRODUCTION_ANALYTICS / f"{STATE_CODE}_Exposure_Rate_Comparison.png", dpi=300, bbox_inches='tight')
plt.close(fig2)


# ==============================================================================
# --- 6. Data Engineering and Spatial Exports ---
# ==============================================================================
print("\n4) Calculating Final Model Metrics...")

led_df['expected_year_built'] = np.round(np.sum(prob_matrix * YEARS, axis=1)).astype(int)
led_df['map_year_built'] = YEARS[np.argmax(prob_matrix, axis=1)]

# UPDATE 5: Implement the HSF Tiers of Confidence Tagging (Building Level)
print("   -> Applying Confidence Tiers...")
# All Probabilistic results are inherently Tier 3.
led_df['Confidence_Tier'] = 'Tier 3 (Modeled - Bayesian Inference)'

# For data transparency, we map the parent pixel's Tier onto the building so researchers know its foundation
parent_tiers = det_matrix[['HISDAC_id', 'Confidence_Tier']].rename(columns={'Confidence_Tier': 'Parent_Pixel_Tier'})
led_df = led_df.merge(parent_tiers, on='HISDAC_id', how='left')

print("5) Saving Master Parquet Engine (with probability arrays)...")
out_parquet = PRODUCTION_SET_PATH / f"{STATE_CODE}_LED_Monte_Carlo_Engine.parquet"
led_df.to_parquet(out_parquet, index=False)


print("6) Preparing Spatial Exports...")

# --- EXPORT A: Probabilistic Model (Building Level Points) ---
print(f"   -> Exporting Probabilistic Building Points...")
gis_df_prob = led_df.drop(columns=['prob_distribution'])
gis_df_prob['geometry'] = led_gdf['geometry'].values
gdf_prob = gpd.GeoDataFrame(gis_df_prob, geometry='geometry', crs=led_gdf.crs)

prob_gpkg = PRODUCTION_SET_PATH / f"{STATE_CODE}_Probabilistic_Building_Points.gpkg"
gdf_prob.to_file(prob_gpkg, driver='GPKG')

# --- EXPORT B: Deterministic Model (Pixel Level Centroids) ---
print(f"   -> Exporting Deterministic Pixel Centroids...")
det_spatial = det_matrix.merge(spatial_anchor[['HISDAC_id', 'geometry']], on='HISDAC_id', how='left')
gdf_det = gpd.GeoDataFrame(det_spatial, geometry='geometry', crs=spatial_anchor.crs)
gdf_det['geometry'] = gdf_det.geometry.centroid

det_gpkg = PRODUCTION_SET_PATH / f"{STATE_CODE}_Deterministic_Pixel_Centroids.gpkg"
gdf_det.to_file(det_gpkg, driver='GPKG')

print(f"\n=== SCRIPT 2 COMPLETE FOR {STATE_CODE} ===")