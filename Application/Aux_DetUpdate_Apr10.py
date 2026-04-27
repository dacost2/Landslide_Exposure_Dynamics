# %% [markdown]
# # Deterministic Engine Fast-Updater (National Batch)
# #### Daniel Acosta-Reyes
# %%
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# --- 1. Setup Paths & State List ---
DATA_PATH = Path("/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics/Data")
ANALYSIS_PATH = Path("/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics/Analysis")

STATES_LOWER_48 = [
    'AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'ID', 'IL', 'IN', 'IA', 
    'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 
    'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 
    'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
]

YEARS = np.arange(1920, 2025, 5)
hazard_classes = ['high', 'moderate', 'low', 'none']
color_map = {'high': '#d62728', 'moderate': '#ff7f0e', 'low': '#f1c40f', 'none': '#7f8c8d'}

print("=== Starting Deterministic National Batch Update ===")

for STATE_CODE in STATES_LOWER_48:
    PROD_PATH = DATA_PATH / "Production_set" / STATE_CODE
    ANALYTICS_PATH = ANALYSIS_PATH / "Production_analytics" / STATE_CODE
    
    # Check if this state has been processed by Script 1
    master_file = PROD_PATH / f"{STATE_CODE}_Master_Spatiotemporal_Matrix.parquet"
    if not master_file.exists():
        continue
        
    print(f"\nProcessing {STATE_CODE}...")
    ANALYTICS_PATH.mkdir(parents=True, exist_ok=True)
    
    # --- 2. Load Datasets ---
    master_df = pd.read_parquet(master_file)
    
    # We only need the LED file to get the susc_class counts per pixel
    led_df = pd.read_parquet(PROD_PATH / f"{STATE_CODE}_LED_Joined_Buildings.parquet")
    spatial_anchor = gpd.read_file(PROD_PATH / f"{STATE_CODE}_HISDAC_Spatial_Anchor.gpkg")

    led_df['susc_class'] = led_df['susc_class'].astype(str).str.lower().str.strip()
    
    # --- 3. Deterministic Math (With Smooth Fallback) ---
    cumulative_exposure_D = {c: np.zeros(len(YEARS)) for c in hazard_classes}
    
    led_exposed_counts = led_df.groupby(['HISDAC_id', 'susc_class']).size().unstack(fill_value=0)
    det_matrix = master_df.join(led_exposed_counts, on='HISDAC_id', how='left')

    for c in hazard_classes:
        if c not in det_matrix.columns: det_matrix[c] = 0
    det_matrix[hazard_classes] = det_matrix[hazard_classes].fillna(0)

    det_matrix['Total_LED_2020'] = det_matrix[hazard_classes].sum(axis=1)
    det_matrix['Total_HISDAC'] = det_matrix['C_BUPL2020'].fillna(0)
    det_matrix['Cumulative_LED_Fallback'] = 0

    for y_idx, y in enumerate(YEARS):
        cumulative_hisdac = det_matrix[f'C_BUPL{y}'].fillna(0).values
        
        if f'LED_{y}' in det_matrix.columns:
            det_matrix['Cumulative_LED_Fallback'] += det_matrix[f'LED_{y}'].fillna(0)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_hisdac = np.where(det_matrix['Total_HISDAC'] > 0, cumulative_hisdac / det_matrix['Total_HISDAC'], 0)
            ratio_led = np.where(det_matrix['Total_LED_2020'] > 0, det_matrix['Cumulative_LED_Fallback'] / det_matrix['Total_LED_2020'], 0)
                                 
        growth_ratio = np.where(det_matrix['Total_HISDAC'] > 0, ratio_hisdac, ratio_led)
        
        if y == 2020:
            growth_ratio = 1.0
            
        for c in hazard_classes:
            exposed_in_pixel = det_matrix[c].values * growth_ratio
            cumulative_exposure_D[c][y_idx] = exposed_in_pixel.sum()

    # --- 4. Generate Updated Analytics Plots ---
    total_built_D = np.zeros(len(YEARS))
    for c in hazard_classes: total_built_D += cumulative_exposure_D[c]

    rate_D = {c: np.zeros(len(YEARS)) for c in hazard_classes}
    for c in hazard_classes:
        with np.errstate(divide='ignore', invalid='ignore'):
            rate_D[c] = np.where(total_built_D > 0, (cumulative_exposure_D[c] / total_built_D) * 100, 0)

    # Plot 1: Cumulative
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    fig1.suptitle(f"Cumulative Landslide Exposure: Deterministic Baseline ({STATE_CODE})", fontsize=16, fontweight='bold')
    for c in ['high', 'moderate', 'low']:
        ax1.plot(YEARS, cumulative_exposure_D[c], color=color_map[c], linestyle='-', linewidth=3, marker='o', label=f"{c.capitalize()} (Deterministic)")
    ax1.set_ylabel("Total Cumulative Buildings Exposed")
    ax1.set_xlabel("Year")
    ax1.set_xticks(YEARS)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper left')
    fig1.savefig(ANALYTICS_PATH / f"{STATE_CODE}_Deterministic_Cumulative.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # Plot 2: Rates
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    fig2.suptitle(f"Landslide Exposure Rate E(t): Deterministic Baseline ({STATE_CODE})", fontsize=16, fontweight='bold')
    for c in ['high', 'moderate', 'low']:
        ax2.plot(YEARS, rate_D[c], color=color_map[c], linestyle='-', linewidth=3, marker='o', label=f"{c.capitalize()} Rate")
    ax2.set_ylabel("Exposure Rate (%)")
    ax2.set_xlabel("Year")
    ax2.set_xticks(YEARS)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper left')
    fig2.savefig(ANALYTICS_PATH / f"{STATE_CODE}_Deterministic_Rate.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # --- 5. Export Updated Deterministic GPKG ---
    det_spatial = det_matrix.merge(spatial_anchor[['HISDAC_id', 'geometry']], on='HISDAC_id', how='left')
    gdf_det = gpd.GeoDataFrame(det_spatial, geometry='geometry', crs=spatial_anchor.crs)
    gdf_det['geometry'] = gdf_det.geometry.centroid
    
    det_gpkg = PROD_PATH / f"{STATE_CODE}_Deterministic_Pixel_Centroids.gpkg"
    gdf_det.to_file(det_gpkg, driver='GPKG')
    
    print(f"   -> Updated GPKG and Charts saved.")

print("\n=== BATCH UPDATE COMPLETE ===")