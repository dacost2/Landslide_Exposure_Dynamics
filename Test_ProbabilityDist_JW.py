# %%
'''Deterministic vs Probabilistic Exposure Trajectories'''
import pandas as pd
import numpy as np
from scipy.stats import norm
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- 1. Setup & Load Datasets ---
DATA_PATH = Path("/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics/Data")
TESTING_SET_PATH = DATA_PATH / "Testing_Set"
state_code = "WA"
YEARS = np.arange(1920, 2025, 5)

print("1) Loading Datasets...")
master_df = pd.read_parquet(TESTING_SET_PATH / f"{state_code}_Master_Spatiotemporal_Matrix.parquet")
led_df = pd.read_parquet(TESTING_SET_PATH / f"{state_code}_LED_Joined_Buildings.parquet")

# Ensure susc_class is clean
if 'susc_class' not in led_df.columns:
    raise KeyError("Column 'susc_class' is missing from led_df.")
led_df['susc_class'] = led_df['susc_class'].astype(str).str.lower().str.strip()
hazard_classes = ['none', 'low', 'moderate', 'high']

master_idx = master_df.set_index('HISDAC_id')

# --- 2. User Input Selection ---
print("\n--- Model Selection ---")
print("P: Probabilistic (Bayesian Engine)")
print("D: Deterministic (Area-Level Supervisor Method)")
print("all: Run Both and Compare")
mode = input("Choose method ('P', 'D', or 'all'): ").strip().lower()

# Dictionaries to store the final exposure trajectories
cumulative_exposure_P = {c: np.zeros(len(YEARS)) for c in hazard_classes}
cumulative_exposure_D = {c: np.zeros(len(YEARS)) for c in hazard_classes}

# --- 3. Probabilistic Execution ---
if mode in ['p', 'all']:
    print("\n2a) Running Probabilistic Engine...")
    
    def generate_probability_array(row):
        hid = row['HISDAC_id']
        if hid not in master_idx.index:
            prob_array = np.zeros(21)
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
        
        prob_array = np.zeros(21)
        
        if pd.isna(fbuy) or np.sum(hisdac_bupl) == 0:
            year_idx = np.where(YEARS == led_year)[0][0]
            prob_array[year_idx] = 1.0
            return prob_array

        if density > 3.0:
            if stories > 3:
                hisdac_bupr = np.array([pixel[f'D_BUPR{y}'] for y in YEARS])
                if np.sum(hisdac_bupr) > 0: return hisdac_bupr / np.sum(hisdac_bupr)
                return np.ones(21) / 21
            else:
                base_dist = hisdac_bupl / np.sum(hisdac_bupl)
                gaussian_weights = norm.pdf(YEARS, loc=led_year, scale=10)
                prob_array = base_dist * gaussian_weights
                return prob_array / np.sum(prob_array) if np.sum(prob_array) > 0 else base_dist

        if source_nsi == 'nsi_estimated':
            recent_idx = slice(16, 21)
            deficits = np.maximum(0, hisdac_bupl[recent_idx] - led_counts[recent_idx])
            if np.sum(deficits) > 0:
                prob_array[recent_idx] = deficits / np.sum(deficits)
                return prob_array

        base_dist = hisdac_bupl / np.sum(hisdac_bupl)
        gaussian_weights = norm.pdf(YEARS, loc=led_year, scale=5)
        prob_array = base_dist * gaussian_weights
        if np.sum(prob_array) > 0: return prob_array / np.sum(prob_array)
        return np.ones(21) / 21

    tqdm.pandas(desc="Calculating Probabilities")
    led_df['prob_distribution'] = led_df.progress_apply(generate_probability_array, axis=1)
    
    # Calculate Expected Values
    prob_matrix = np.stack(led_df['prob_distribution'].values)
    # Convert periodic probabilities into cumulative probabilities to match the deterministic logic
    cum_prob_matrix = np.cumsum(prob_matrix, axis=1)
    
    for c in hazard_classes:
        mask = led_df['susc_class'] == c
        if mask.sum() > 0:
            cumulative_exposure_P[c] = np.sum(cum_prob_matrix[mask], axis=0)

# %%
# ==============================================================================
# --- 4. Deterministic Execution (UPDATED) ---
# ==============================================================================
if mode in ['d', 'all']:
    print("\n2b) Running Deterministic Engine (Area-Level Method)...")
    
    # Step 1: Calculate ABSOLUTE LED exposure counts per HISDAC pixel in 2020 (Not fractions)
    print("   -> Calculating 2020 LED Exposure Counts...")
    led_exposed_counts = led_df.groupby(['HISDAC_id', 'susc_class']).size().unstack(fill_value=0)
    
    # Step 2: Merge counts onto the HISDAC master matrix
    det_matrix = master_df.join(led_exposed_counts, on='HISDAC_id', how='left')
    for c in hazard_classes:
        if c not in det_matrix.columns:
            det_matrix[c] = 0
    det_matrix[hazard_classes] = det_matrix[hazard_classes].fillna(0)

    # Calculate Total HISDAC built up to 2020 to act as our denominator
    bupl_cols = [f'D_BUPL{y}' for y in YEARS]
    det_matrix['Total_HISDAC'] = det_matrix[bupl_cols].sum(axis=1)

    # Step 3: Turn back the clock (Distribute LED counts using HISDAC growth ratios)
    print("   -> Distributing LED counts using HISDAC growth ratios...")
    cumulative_hisdac = np.zeros(len(det_matrix))
    
    for y_idx, y in enumerate(YEARS):
        cumulative_hisdac += det_matrix[f'D_BUPL{y}'].fillna(0).values
        
        # Calculate the ratio of the built environment at year 't' relative to 2020 (0.0 to 1.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            growth_ratio = np.where(det_matrix['Total_HISDAC'] > 0,
                                    cumulative_hisdac / det_matrix['Total_HISDAC'],
                                    0)
        
        # Force the 2020 ratio to exactly 1.0 (Resolves pixels where HISDAC was empty but LED had buildings)
        if y == 2020:
            growth_ratio = np.where(det_matrix['Total_HISDAC'] == 0, 1.0, growth_ratio)
            
        # Step 4: Multiply the fixed 2020 LED count by the historical growth ratio
        for c in hazard_classes:
            exposed_in_pixel = det_matrix[c].values * growth_ratio
            cumulative_exposure_D[c][y_idx] = exposed_in_pixel.sum()

# --- 5. Visualization (If Comparing Both) ---
if mode == 'all':
    print("\n3) Generating Comparison Plot...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(f"Cumulative Landslide Exposure: Probabilistic vs Deterministic ({state_code})", fontsize=16, fontweight='bold')
    
    color_map = {'high': '#d62728', 'moderate': '#ff7f0e', 'low': '#f1c40f', 'none': '#7f8c8d'}
    
    for c in ['high', 'moderate', 'low']: # Skip 'none' to focus on hazards
        color = color_map[c]
        
        # Plot Probabilistic (Solid Line)
        ax.plot(YEARS, cumulative_exposure_P[c], color=color, linestyle='-', linewidth=3, 
                marker='o', label=f"{c.capitalize()} (Probabilistic)")
        
        # Plot Deterministic (Dashed Line)
        ax.plot(YEARS, cumulative_exposure_D[c], color=color, linestyle='--', linewidth=2, 
                alpha=0.7, label=f"{c.capitalize()} (Deterministic Baseline)")

    ax.set_title("Methodological Comparison: Does assuming static exposure fractions skew history?", fontsize=12)
    ax.set_ylabel("Total Cumulative Buildings Exposed")
    ax.set_xlabel("Year")
    ax.set_xticks(YEARS)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Clean up the legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', ncol=2)
    
    plt.tight_layout()
    plt.show()

print("\nPipeline Complete!")
# %%
# ==============================================================================
# --- 6. MSA Zoom-In: Seattle-Tacoma-Bellevue (UPDATED) ---
# ==============================================================================
import geopandas as gpd

if mode == 'all':
    print("\n--- 4) Running Regional Subset: Seattle-Tacoma-Bellevue MSA ---")
    
    CBSA_PATH = DATA_PATH / "cb_2018_us_cbsa_500k.zip"
    LED_SPATIAL_PATH = TESTING_SET_PATH / f"{state_code}_LED_points-5070.gpkg" 
    
    print("   -> Loading and filtering CBSA Shapefile...")
    cbsa_gdf = gpd.read_file(f"zip://{CBSA_PATH}")
    seattle_msa = cbsa_gdf[cbsa_gdf['NAME'] == 'Seattle-Tacoma-Bellevue, WA'].copy()
    
    print("   -> Loading spatial footprints and intersecting...")
    led_spatial = gpd.read_file(LED_SPATIAL_PATH)
    seattle_msa = seattle_msa.to_crs(led_spatial.crs)
    seattle_buildings = gpd.sjoin(led_spatial, seattle_msa, how="inner", predicate="intersects")
    
    led_seattle = led_df.loc[led_df.index.isin(seattle_buildings.index)].copy()
    seattle_hids = led_seattle['HISDAC_id'].unique()
    det_seattle = det_matrix[det_matrix.index.isin(seattle_hids)].copy()

    # --- Re-Calculate Probabilistic (Seattle) ---
    print("   -> Re-aggregating Probabilistic Engine for Seattle MSA...")
    cum_exp_P_sea = {c: np.zeros(len(YEARS)) for c in hazard_classes}
    prob_matrix_sea = np.stack(led_seattle['prob_distribution'].values)
    cum_prob_matrix_sea = np.cumsum(prob_matrix_sea, axis=1)
    
    for c in hazard_classes:
        mask = led_seattle['susc_class'] == c
        if mask.sum() > 0:
            cum_exp_P_sea[c] = np.sum(cum_prob_matrix_sea[mask], axis=0)

    # --- Re-Calculate Deterministic (Seattle) ---
    print("   -> Re-aggregating Deterministic Engine for Seattle MSA...")
    cum_exp_D_sea = {c: np.zeros(len(YEARS)) for c in hazard_classes}
    cumulative_hisdac_sea = np.zeros(len(det_seattle))
    
    for y_idx, y in enumerate(YEARS):
        cumulative_hisdac_sea += det_seattle[f'D_BUPL{y}'].fillna(0).values
        
        with np.errstate(divide='ignore', invalid='ignore'):
            growth_ratio_sea = np.where(det_seattle['Total_HISDAC'] > 0,
                                        cumulative_hisdac_sea / det_seattle['Total_HISDAC'],
                                        0)
        if y == 2020:
            growth_ratio_sea = np.where(det_seattle['Total_HISDAC'] == 0, 1.0, growth_ratio_sea)
            
        for c in hazard_classes:
            exposed_in_pixel_sea = det_seattle[c].values * growth_ratio_sea
            cum_exp_D_sea[c][y_idx] = exposed_in_pixel_sea.sum()

    # --- Plot the Regional Comparison ---
    # (The plotting code remains exactly the same as the previous script)
    print("   -> Generating Regional Comparison Plot...")
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(f"Cumulative Landslide Exposure: Seattle-Tacoma-Bellevue MSA", fontsize=16, fontweight='bold')
    
    color_map = {'high': '#d62728', 'moderate': '#ff7f0e', 'low': '#f1c40f', 'none': '#7f8c8d'}
    
    for c in ['high', 'moderate', 'low']:
        color = color_map[c]
        ax.plot(YEARS, cum_exp_P_sea[c], color=color, linestyle='-', linewidth=3, 
                marker='o', label=f"{c.capitalize()} (Probabilistic)")
        ax.plot(YEARS, cum_exp_D_sea[c], color=color, linestyle='--', linewidth=2, 
                alpha=0.7, label=f"{c.capitalize()} (Deterministic Baseline)")

    ax.set_title("Regional Methodological Comparison: The Urban Core Dynamics", fontsize=12)
    ax.set_ylabel("Total Cumulative Buildings Exposed")
    ax.set_xlabel("Year")
    ax.set_xticks(YEARS)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', ncol=2)
    
    plt.tight_layout()
    plt.show()

print("\nAll tasks completed!")