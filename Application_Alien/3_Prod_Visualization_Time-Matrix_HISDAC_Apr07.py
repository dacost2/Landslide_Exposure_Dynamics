# %% [markdown]
# # Visualization and Analytics Engine (Production)
# #### Daniel Acosta-Reyes
# April 07, 2026
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from pathlib import Path
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

# Suppress standard pandas/numpy division by zero warnings for clean logs
warnings.filterwarnings('ignore')

# --- 1. Argparse Setup for Master Controller ---
parser = argparse.ArgumentParser(description="Run Analytics and Visualization for a specific state.")
parser.add_argument("--state", type=str, required=True, help="2-letter state code (e.g., WA)")
args, unknown = parser.parse_known_args()
STATE_CODE = args.state.upper()

# --- 2. Setup Paths ---
DATA_PATH = Path(r"C:\Users\danie\OneDrive - UW\0 - DA General Exam\Paper 2 - Temporal Dynamics\Data")
PRODUCTION_SET_PATH = DATA_PATH / "Production_set" / STATE_CODE
ANALYSIS_PATH = DATA_PATH.parent / "Analysis"
PRODUCTION_ANALYTICS = ANALYSIS_PATH / "Production_analytics" / STATE_CODE

PRODUCTION_ANALYTICS.mkdir(parents=True, exist_ok=True)
CBSA_PATH = DATA_PATH / "cb_2018_us_cbsa_500k.zip"
YEARS = np.arange(1920, 2025, 5)

print(f"=== Starting Analytics Engine for {STATE_CODE} ===")

# --- 3. Load Data ---
print("1) Loading Production Datasets...")
master_df = pd.read_parquet(PRODUCTION_SET_PATH / f"{STATE_CODE}_Master_Spatiotemporal_Matrix.parquet")
led_engine = pd.read_parquet(PRODUCTION_SET_PATH / f"{STATE_CODE}_LED_Monte_Carlo_Engine.parquet")
led_points = gpd.read_file(PRODUCTION_SET_PATH / f"{STATE_CODE}_Probabilistic_Building_Points.gpkg", columns=['HISDAC_id', 'geometry'])
det_centroids = gpd.read_file(PRODUCTION_SET_PATH / f"{STATE_CODE}_Deterministic_Pixel_Centroids.gpkg")

# --- Helper: Bin years to Semi-Decades ---
def bin_to_semi_decade(series):
    y = pd.to_numeric(series, errors="coerce")
    sd = ((y - 1) // 5 + 1) * 5
    return np.clip(sd, 1920, 2020)

led_engine['expected_bin'] = bin_to_semi_decade(led_engine['expected_year_built'])
led_engine['map_bin'] = bin_to_semi_decade(led_engine['map_year_built'])

# ==============================================================================
# --- 4. Validation Metrics Matrix (Req 3) ---
# ==============================================================================
print("2) Calculating Temporal Validation Metrics...")

hisdac_cols = [f"D_BUPL{y}" for y in YEARS]
hisdac_counts = master_df[hisdac_cols].sum().values

# Extract counts for the three LED variations
raw_counts = led_engine['semi_decade'].value_counts().reindex(YEARS, fill_value=0).values
exp_counts = led_engine['expected_bin'].value_counts().reindex(YEARS, fill_value=0).values
map_counts = led_engine['map_bin'].value_counts().reindex(YEARS, fill_value=0).values

# A) Volume Check
# hisdac_total_2020 = hisdac_counts.sum()
# led_total_2020 = len(led_engine)
# abs_diff = led_total_2020 - hisdac_total_2020
# pct_diff = (abs_diff / hisdac_total_2020) * 100 if hisdac_total_2020 > 0 else np.nan

# A) Volume Check -- UPDATED APR 09 -- 
# Fix: Use the true cumulative snapshot for 2020 to capture the pre-1920 backlog
hisdac_total_2020 = master_df['C_BUPL2020'].sum()
led_total_2020 = len(led_engine)

abs_diff = led_total_2020 - hisdac_total_2020
pct_diff = (abs_diff / hisdac_total_2020) * 100 if hisdac_total_2020 > 0 else np.nan

# FBUY Calculation Function
def calc_fbuy_pass_rate(year_col):
    first_year = led_engine.groupby('HISDAC_id')[year_col].min().rename('first_led')
    df = master_df[['HISDAC_id', 'FBUY']].dropna().merge(first_year, on='HISDAC_id', how='inner')
    passes = df['first_led'] >= (df['FBUY'] - 5)
    return (passes.sum() / len(df)) * 100 if len(df) > 0 else np.nan

fbuy_raw = calc_fbuy_pass_rate('semi_decade')
fbuy_exp = calc_fbuy_pass_rate('expected_bin')
fbuy_map = calc_fbuy_pass_rate('map_bin')

# Statistical Metrics Function
def get_stats(led_arr, hisdac_arr):
    r_val = np.corrcoef(hisdac_arr, led_arr)[0, 1]
    r_squared = r_val ** 2 if not np.isnan(r_val) else np.nan
    rmse = np.sqrt(mean_squared_error(hisdac_arr, led_arr))
    mae = mean_absolute_error(hisdac_arr, led_arr)
    return r_squared, r_val, rmse, mae

r2_raw, r_raw, rmse_raw, mae_raw = get_stats(raw_counts, hisdac_counts)
r2_exp, r_exp, rmse_exp, mae_exp = get_stats(exp_counts, hisdac_counts)
r2_map, r_map, rmse_map, mae_map = get_stats(map_counts, hisdac_counts)

# Compile DataFrame
metrics_data = {
    'Metric': ['HISDAC Total 2020', 'LED Total 2020', 'Absolute Difference', '% Difference'],
    'Value': [hisdac_total_2020, led_total_2020, abs_diff, pct_diff],
    'Raw_vs_HISDAC': [np.nan, np.nan, np.nan, np.nan],
    'Expected_vs_HISDAC': [np.nan, np.nan, np.nan, np.nan],
    'MAP_vs_HISDAC': [np.nan, np.nan, np.nan, np.nan]
}
metrics_df = pd.DataFrame(metrics_data)

stats_df = pd.DataFrame({
    'Metric': ['R^2', 'Pearson_r', 'RMSE', 'MAE', '% Passing FBUY'],
    'Value': [np.nan] * 5,
    'Raw_vs_HISDAC': [r2_raw, r_raw, rmse_raw, mae_raw, fbuy_raw],
    'Expected_vs_HISDAC': [r2_exp, r_exp, rmse_exp, mae_exp, fbuy_exp],
    'MAP_vs_HISDAC': [r2_map, r_map, rmse_map, mae_map, fbuy_map]
})

final_metrics = pd.concat([metrics_df, stats_df], ignore_index=True)
final_metrics.to_csv(PRODUCTION_ANALYTICS / f"{STATE_CODE}_Metrics_Summary.csv", index=False)

# ==============================================================================
# --- 5. The Tri-Curve Plot (Req 4) ---
# ==============================================================================
print("3) Generating Tri-Curve Plot...")
fig, ax = plt.subplots(figsize=(14, 6))
width = 1.6

ax.bar(YEARS - width/2, raw_counts, width=width, label='LED Raw (MAUP Clumping)', color='#ff7f0e', alpha=0.5)
ax.bar(YEARS + width/2, exp_counts, width=width, label='LED Expected (Probabilistic)', color='#2ca02c', alpha=0.8)
ax.plot(YEARS, hisdac_counts, color='blue', marker='o', linestyle='-', linewidth=2.5, markersize=6, label='Raw HISDAC (Tax Records Baseline)')

ax.set_title(f'Temporal Reconstruction Fit: {STATE_CODE}', fontsize=14, fontweight='bold')
ax.set_ylabel('New Buildings Added')
ax.set_xticks(YEARS)
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(loc='upper left')

fig.tight_layout()
fig.savefig(PRODUCTION_ANALYTICS / f"{STATE_CODE}_Temporal_TriCurve.png", dpi=300)
plt.close(fig)

# ==============================================================================
# --- 6. The Exposure Story Core Logic (Req 5) ---
# ==============================================================================
print("4) Processing 'The Story' DataFrames...")
CHECK_YEARS = [1920, 1940, 1960, 1980, 2000, 2020]
HAZARDS = ['high', 'moderate', 'low', 'none']

def calculate_story(led_subset, det_subset, region_name):
    """Calculates Cumulative and Marginal exposure for Probabilistic and Deterministic models."""
    story_records = []
    
    # --- PROBABILISTIC (Building Level) ---
    # Fix: Use map_bin to guarantee consistency between the regional tables and the spatial maps
    prob_counts = led_subset.groupby(['map_bin', 'susc_class']).size().unstack(fill_value=0).reindex(YEARS, fill_value=0)
    for h in HAZARDS:
        if h not in prob_counts.columns: prob_counts[h] = 0
            
    prob_cum = prob_counts[HAZARDS].cumsum()
    prob_cum['Total'] = prob_cum.sum(axis=1)
    
    prob_marg = prob_counts[HAZARDS]
    prob_marg['Total'] = prob_marg.sum(axis=1)

    # --- DETERMINISTIC (Pixel Level Ratios - SNAPSHOT FIX) ---
    det_cum = {h: np.zeros(len(YEARS)) for h in HAZARDS}
    
    # Use the 2020 Cumulative Snapshot as the absolute historical denominator
    det_subset['Total_HISDAC'] = det_subset['C_BUPL2020'].fillna(0)
    
    for i, y in enumerate(YEARS):
        # Grab the specific year's cumulative snapshot directly
        cum_hisdac = det_subset[f'C_BUPL{y}'].fillna(0).values
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(det_subset['Total_HISDAC'] > 0, cum_hisdac / det_subset['Total_HISDAC'], 0)
            
        if y == 2020: 
            ratio = np.where(det_subset['Total_HISDAC'] == 0, 1.0, ratio)
            
        for h in HAZARDS:
            det_cum[h][i] = (det_subset[h].values * ratio).sum()
            
    det_cum_df = pd.DataFrame(det_cum, index=YEARS)
    det_cum_df['Total'] = det_cum_df.sum(axis=1)
    
    # Calculate Marginal via diff()
    det_marg_df = det_cum_df.diff().fillna(det_cum_df.iloc[0])
    
    # --- Extract Check Years ---
    for y in CHECK_YEARS:
        for h in HAZARDS:
            # Probabilistic
            c_vol_p = prob_cum.loc[y, h]
            c_tot_p = prob_cum.loc[y, 'Total']
            m_vol_p = prob_marg.loc[y, h]
            m_tot_p = prob_marg.loc[y, 'Total']
            
            story_records.append({
                'Region': region_name, 'Year': y, 'Method': 'Probabilistic', 'Hazard': h,
                'Cum_Vol': c_vol_p, 'Cum_Rate_%': (c_vol_p/c_tot_p*100) if c_tot_p > 0 else 0,
                'Marginal_Vol': m_vol_p, 'Marginal_Rate_%': (m_vol_p/m_tot_p*100) if m_tot_p > 0 else 0
            })
            
            # Deterministic
            c_vol_d = det_cum_df.loc[y, h]
            c_tot_d = det_cum_df.loc[y, 'Total']
            m_vol_d = det_marg_df.loc[y, h]
            m_tot_d = det_marg_df.loc[y, 'Total']
            
            story_records.append({
                'Region': region_name, 'Year': y, 'Method': 'Deterministic', 'Hazard': h,
                'Cum_Vol': c_vol_d, 'Cum_Rate_%': (c_vol_d/c_tot_d*100) if c_tot_d > 0 else 0,
                'Marginal_Vol': m_vol_d, 'Marginal_Rate_%': (m_vol_d/m_tot_d*100) if m_tot_d > 0 else 0
            })
            
    return pd.DataFrame(story_records)

# A) State Story
state_story_df = calculate_story(led_engine, det_centroids, f"State: {STATE_CODE}")
state_story_df.to_csv(PRODUCTION_ANALYTICS / f"{STATE_CODE}_State_Exposure_Story.csv", index=False)

# B) CBSA Story
print("5) Spatially Joining CBSAs and Generating Metro Stories...")
cbsa_gdf = gpd.read_file(f"zip://{CBSA_PATH}")

# Project to match points
cbsa_gdf = cbsa_gdf.to_crs(led_points.crs)

# Identify which CBSAs intersect this state's buildings
led_with_cbsa = gpd.sjoin(led_points, cbsa_gdf[['NAME', 'geometry']], how='inner', predicate='intersects')
cbsa_names = led_with_cbsa['NAME'].unique()

if len(cbsa_names) > 0:
    det_centroids_crs = det_centroids.to_crs(led_points.crs)
    det_with_cbsa = gpd.sjoin(det_centroids_crs, cbsa_gdf[['NAME', 'geometry']], how='inner', predicate='intersects')
    
    cbsa_story_dfs = []
    for cbsa in cbsa_names:
        # Filter datasets via index/HISDAC matching
        cbsa_led_indices = led_with_cbsa[led_with_cbsa['NAME'] == cbsa].index
        cbsa_led_subset = led_engine.loc[cbsa_led_indices]
        
        cbsa_det_subset = det_with_cbsa[det_with_cbsa['NAME'] == cbsa].copy()
        
        # Recalculate story for this metro
        if len(cbsa_led_subset) > 0:
            cbsa_story = calculate_story(cbsa_led_subset, cbsa_det_subset, cbsa)
            cbsa_story_dfs.append(cbsa_story)
            
    if cbsa_story_dfs:
        final_cbsa_df = pd.concat(cbsa_story_dfs, ignore_index=True)
        final_cbsa_df.to_csv(PRODUCTION_ANALYTICS / f"{STATE_CODE}_CBSA_Exposure_Story.csv", index=False)
        print(f"   -> Extracted exposure stories for {len(cbsa_names)} Metropolitan Areas.")
else:
    print("   -> No CBSA intersections found for this state.")

print(f"\n=== SCRIPT 3 COMPLETE FOR {STATE_CODE} ===")

# ==============================================================================
# --- 7. Spatial Story Cartography (3x2 CBSA Map Grids) ---
# ==============================================================================
if len(cbsa_names) > 0:
    print("6) Generating 3x2 CBSA Spatial Map Grids (Both Methods)...")
    import matplotlib.colors as mcolors

    # 1. Load State Boundary for visual background context
    STATE_BOUNDARY_PATH = DATA_PATH / "cb_2024_us_state_500k.zip"
    state_boundaries = gpd.read_file(f"zip://{STATE_BOUNDARY_PATH}")
    state_geom = state_boundaries[state_boundaries['STUSPS'] == STATE_CODE].to_crs(cbsa_gdf.crs)
    
    # 2. Filter for High Hazard to calculate GLOBAL min/max for the colorbars
    high_hazard_df = final_cbsa_df[final_cbsa_df['Hazard'] == 'high']
    
    # Calculate global limits for Absolute Volume
    vmin_vol = high_hazard_df['Cum_Vol'].min()
    vmax_vol = high_hazard_df['Cum_Vol'].max()
    if vmin_vol == vmax_vol: vmax_vol = vmin_vol + 1
        
    # Calculate global limits for Exposure Rate
    vmin_rate = high_hazard_df['Cum_Rate_%'].min()
    vmax_rate = high_hazard_df['Cum_Rate_%'].max()
    if vmin_rate == vmax_rate: vmax_rate = vmin_rate + 1

    # 3. Create Map Generation Function
    def generate_3x2_map_grid(metric_col, title_metric_name, cmap, filename_suffix, method_name, vmin, vmax):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"High-Hazard Landslide {title_metric_name} ({method_name} Baseline)\nCBSA: {STATE_CODE}", fontsize=20, fontweight='bold')
        axes = axes.flatten()
        
        # Filter the story data for the specific method we are plotting
        map_story_df = high_hazard_df[high_hazard_df['Method'] == method_name]
        
        for i, year in enumerate(CHECK_YEARS):
            ax = axes[i]
            
            # Plot the state background
            state_geom.plot(ax=ax, color='#e0e0e0', edgecolor='white', linewidth=1.5)
            
            # Filter data for the specific year
            year_data = map_story_df[map_story_df['Year'] == year]
            
            # Merge back to CBSA geometries
            map_data = cbsa_gdf.merge(year_data, left_on='NAME', right_on='Region', how='inner')
            
            if not map_data.empty:
                # Plot the CBSAs using the GLOBAL vmin and vmax
                map_data.plot(column=metric_col, cmap=cmap, ax=ax, 
                              vmin=vmin, vmax=vmax, edgecolor='black', linewidth=0.5)
            
            ax.set_title(f"Year: {year}", fontsize=14, fontweight='bold')
            ax.set_axis_off()
            
        # Add a single, unified colorbar to the figure
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03]) # [left, bottom, width, height]
        fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', label=title_metric_name)
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.93]) # Adjust bounds to fit colorbar and the 2-line title
        
        # Save with the method appended to the filename
        final_filename = f"{STATE_CODE}_{filename_suffix}_{method_name}.png"
        plt.savefig(PRODUCTION_ANALYTICS / final_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

    # 4. Execute the mapping function for both methods
    for method in ['Probabilistic', 'Deterministic']:
        print(f"   -> Rendering {method} Maps...")
        
        generate_3x2_map_grid(
            metric_col='Cum_Vol', 
            title_metric_name='Cumulative Exposed Buildings (Absolute Volume)', 
            cmap='YlOrRd', 
            filename_suffix='CBSA_Map_Grid_AbsoluteVolume',
            method_name=method,
            vmin=vmin_vol,
            vmax=vmax_vol
        )
        
        generate_3x2_map_grid(
            metric_col='Cum_Rate_%', 
            title_metric_name='Exposure Rate (%)', 
            cmap='Reds', 
            filename_suffix='CBSA_Map_Grid_ExposureRate',
            method_name=method,
            vmin=vmin_rate,
            vmax=vmax_rate
        )

print(f"\n=== SCRIPT 3 COMPLETE FOR {STATE_CODE} ===")