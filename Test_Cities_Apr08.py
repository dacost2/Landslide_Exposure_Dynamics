# %%
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import contextily as ctx
from pathlib import Path
from geopy.geocoders import Nominatim
import sys

# --- 1. User Inputs & Automated Geocoding ---
state_code = input("Enter 2-letter State Code (e.g., WA): ").strip().upper()
city_name = input(f"Enter City Name in {state_code} (e.g., Seattle): ").strip().title()

print(f"\n1) Geocoding {city_name}, {state_code}...")
geolocator = Nominatim(user_agent="landslide_exposure_research")
location = geolocator.geocode(f"{city_name}, {state_code}, USA")

if not location:
    print(f"[ERROR] Could not find {city_name}, {state_code}. Please check spelling.")
    sys.exit(1)

# --- 2. Setup Paths ---
DATA_PATH = Path("/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics/Data")
PROD_PATH = DATA_PATH / "Production_set" / state_code
ANALYSIS_PATH = Path("/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics/Analysis")
OUTPUT_PATH = ANALYSIS_PATH / "cities_test_maps"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# File Paths
spatial_anchor_path = PROD_PATH / f"{state_code}_HISDAC_Spatial_Anchor.gpkg"
det_centroids_path = PROD_PATH / f"{state_code}_Deterministic_Pixel_Centroids.gpkg"
prob_points_path = PROD_PATH / f"{state_code}_Probabilistic_Building_Points.gpkg"

# --- 3. Load and Clip Data ---
print("2) Loading and Clipping Data to City Extent...")

# Load just the Spatial Anchor first to establish CRS
spatial_anchor = gpd.read_file(spatial_anchor_path)
data_crs = spatial_anchor.crs

# Create City Square Extent (15km radius converted to a square bounding box)
city_point = gpd.GeoDataFrame(geometry=gpd.points_from_xy([location.longitude], [location.latitude]), crs="EPSG:4326")
city_point = city_point.to_crs(data_crs)
city_extent = gpd.GeoDataFrame(geometry=city_point.buffer(15000).envelope, crs=data_crs) 

# Clip Spatial Anchor (Pixels)
city_pixels = gpd.clip(spatial_anchor, city_extent)

# Load and Clip Deterministic Centroids, then merge back to polygons to plot as pixels
det_centroids = gpd.read_file(det_centroids_path, mask=city_extent)
city_det = city_pixels[['HISDAC_id', 'geometry']].merge(det_centroids.drop(columns='geometry'), on='HISDAC_id', how='inner')

# Load and Clip Probabilistic Points
city_prob = gpd.read_file(prob_points_path, mask=city_extent)

# --- 4. Calculate Deterministic Temporal Volumes (SNAPSHOT FIX) ---
print("3) Calculating Deterministic Historical Volumes...")
hazard_cols = ['high', 'moderate', 'low', 'none']

for c in hazard_cols:
    if c not in city_det.columns: city_det[c] = 0
city_det[hazard_cols] = city_det[hazard_cols].fillna(0)

# Total LED built environment in 2020
city_det['Total_LED_2020'] = city_det[hazard_cols].sum(axis=1)

# NEW: Use the 2020 Cumulative Snapshot as the absolute historical denominator
city_det['Total_HISDAC'] = city_det['C_BUPL2020'].fillna(0)

check_years = [1920, 1940, 1960, 1980, 2000, 2020]

for y in check_years:
    # NEW: Grab the specific year's cumulative snapshot directly! No summing required.
    cum_hisdac = city_det[f'C_BUPL{y}'].fillna(0)
    
    # Calculate Ratio: (Historical Snapshot / 2020 Snapshot)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(city_det['Total_HISDAC'] > 0, cum_hisdac / city_det['Total_HISDAC'], 0)
    
    # FORCE 2020 to exactly 1.0 (100% of LED totals)
    if y == 2020: 
        ratio = np.where(city_det['Total_HISDAC'] == 0, 1.0, ratio)
        
    city_det[f'Det_Total_{y}'] = city_det['Total_LED_2020'] * ratio
    city_det[f'Det_High_{y}'] = city_det['high'] * ratio

    # THE FIX: Replace 0s with NaNs so zero-value pixels become transparent!
    city_det[f'Det_Total_{y}'] = city_det[f'Det_Total_{y}'].replace(0, np.nan)
    city_det[f'Det_High_{y}'] = city_det[f'Det_High_{y}'].replace(0, np.nan)

# --- 4b. Calculate Probabilistic Pixel Volumes ---
print("3b) Aggregating Probabilistic Points to Pixel Level...")

# Create a clean pixel geometry dataframe
city_prob_pixels = city_pixels[['HISDAC_id', 'geometry']].copy()

for y in check_years:
    # Filter the building points directly (No spatial join needed!)
    # py = city_prob[city_prob['expected_year_built'] <= y]

    # To this:
    py = city_prob[city_prob['map_year_built'] <= y]
    
    # Count Total Buildings per pixel
    tot_counts = py.groupby('HISDAC_id').size().rename(f'Prob_Tot_{y}')
    city_prob_pixels = city_prob_pixels.merge(tot_counts, on='HISDAC_id', how='left')
    
    # Count High Hazard Buildings per pixel
    hy = py[py['susc_class'] == 'high']
    high_counts = hy.groupby('HISDAC_id').size().rename(f'Prob_High_{y}')
    city_prob_pixels = city_prob_pixels.merge(high_counts, on='HISDAC_id', how='left')

    # Replace 0s with NaNs so empty pixels don't plot as dark blocks
    city_prob_pixels[f'Prob_Tot_{y}'] = city_prob_pixels[f'Prob_Tot_{y}'].replace(0, np.nan)
    city_prob_pixels[f'Prob_High_{y}'] = city_prob_pixels[f'Prob_High_{y}'].replace(0, np.nan)

# --- 5. Plotting Function ---

bounds = city_extent.total_bounds # [minx, miny, maxx, maxy]

def add_basemap_to_ax(ax):
    """Applies your specified Contextily basemap settings."""
    ax.set_xlim([bounds[0], bounds[2]])
    ax.set_ylim([bounds[1], bounds[3]])
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldShadedRelief, 
                    crs=data_crs.to_string(), zoom=12, alpha=0.6, zorder=1)
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, 
                    crs=data_crs.to_string(), zoom=12, alpha=0.9, zorder=0)
    ax.set_axis_off()

# ==============================================================================
# --- PLOT 1: DETERMINISTIC (3x4 GRID) ---
# ==============================================================================
print(f"4) Rendering Deterministic 3x4 Grid for {city_name}...")
fig1, axes1 = plt.subplots(3, 4, figsize=(20, 16))
fig1.suptitle(f"Deterministic Exposure Baseline: {city_name}, {state_code}", fontsize=24, fontweight='bold', y=0.96)

vmax_tot_det = city_det['Det_Total_2020'].max()
vmax_high_det = city_det['Det_High_2020'].max()

for i, y in enumerate(check_years):
    row = i // 2
    col_tot = (i % 2) * 2
    col_high = (i % 2) * 2 + 1
    
    ax_tot = axes1[row, col_tot]
    ax_high = axes1[row, col_high]

    # Calculate Metrics
    tot_val = city_det[f'Det_Total_{y}'].sum()
    high_val = city_det[f'Det_High_{y}'].sum()
    pct_val = (high_val / tot_val * 100) if tot_val > 0 else 0

    # COLUMN 1: Total Accumulation
    add_basemap_to_ax(ax_tot)
    city_det.plot(column=f'Det_Total_{y}', ax=ax_tot, cmap='YlGnBu', vmin=0, vmax=vmax_tot_det, alpha=0.8, zorder=2)
    ax_tot.set_title(f"{y} Total Buildings\nAccumulated: {tot_val:,.0f}", fontsize=14)

    # COLUMN 2: High Exposure
    add_basemap_to_ax(ax_high)
    city_det.plot(column=f'Det_High_{y}', ax=ax_high, cmap='YlOrRd', vmin=0, vmax=vmax_high_det, alpha=0.8, zorder=2)
    ax_high.set_title(f"{y} High Susceptibility\nAccumulated: {high_val:,.0f} ({pct_val:.1f}%)", fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig1.savefig(OUTPUT_PATH / f"{state_code}_{city_name}_Deterministic_Grid.png", dpi=300, bbox_inches='tight')
plt.close(fig1)

# ==============================================================================
# --- PLOT 2: PROBABILISTIC POINTS (3x4 GRID) ---
# ==============================================================================
print(f"5) Rendering Probabilistic Points 3x4 Grid for {city_name}...")
fig2, axes2 = plt.subplots(3, 4, figsize=(20, 16))
fig2.suptitle(f"Probabilistic Exposure (Building-Level): {city_name}, {state_code}", fontsize=24, fontweight='bold', y=0.96)

for i, y in enumerate(check_years):
    row = i // 2
    col_tot = (i % 2) * 2
    col_high = (i % 2) * 2 + 1
    
    ax_tot = axes2[row, col_tot]
    ax_high = axes2[row, col_high]

    # points_y = city_prob[city_prob['expected_year_built'] <= y]
    # To this:
    points_y = city_prob[city_prob['map_year_built'] <= y]
    high_points_y = points_y[points_y['susc_class'] == 'high']

    # Calculate Metrics
    tot_val = len(points_y)
    high_val = len(high_points_y)
    pct_val = (high_val / tot_val * 100) if tot_val > 0 else 0

    # COLUMN 1: Total Accumulation
    add_basemap_to_ax(ax_tot)
    if not points_y.empty:
        points_y.plot(ax=ax_tot, color='#2c3e50', markersize=0.5, alpha=0.5, zorder=2)
    ax_tot.set_title(f"{y} Total Buildings\nAccumulated: {tot_val:,.0f}", fontsize=14)

    # COLUMN 2: High Exposure
    add_basemap_to_ax(ax_high)
    if not high_points_y.empty:
        high_points_y.plot(ax=ax_high, color='#c0392b', markersize=0.5, alpha=0.8, zorder=2)
    ax_high.set_title(f"{y} High Susceptibility\nAccumulated: {high_val:,.0f} ({pct_val:.1f}%)", fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig2.savefig(OUTPUT_PATH / f"{state_code}_{city_name}_Probabilistic_Pts_Grid.png", dpi=300, bbox_inches='tight')
plt.close(fig2)

# ==============================================================================
# --- PLOT 3: PROBABILISTIC PIXELS (3x4 GRID) ---
# ==============================================================================
print(f"6) Rendering Probabilistic Pixel 3x4 Grid for {city_name}...")
fig3, axes3 = plt.subplots(3, 4, figsize=(20, 16))
fig3.suptitle(f"Probabilistic Exposure (Aggregated to 250m Pixels): {city_name}, {state_code}", fontsize=24, fontweight='bold', y=0.96)

vmax_tot_prob = city_prob_pixels['Prob_Tot_2020'].max()
vmax_high_prob = city_prob_pixels['Prob_High_2020'].max()

for i, y in enumerate(check_years):
    row = i // 2
    col_tot = (i % 2) * 2
    col_high = (i % 2) * 2 + 1
    
    ax_tot = axes3[row, col_tot]
    ax_high = axes3[row, col_high]

    # Calculate Metrics
    tot_val = city_prob_pixels[f'Prob_Tot_{y}'].sum()
    high_val = city_prob_pixels[f'Prob_High_{y}'].sum()
    pct_val = (high_val / tot_val * 100) if tot_val > 0 else 0

    # COLUMN 1: Total Accumulation
    add_basemap_to_ax(ax_tot)
    city_prob_pixels.plot(column=f'Prob_Tot_{y}', ax=ax_tot, cmap='YlGnBu', vmin=0, vmax=vmax_tot_prob, alpha=0.8, zorder=2)
    ax_tot.set_title(f"{y} Total Buildings\nAccumulated: {tot_val:,.0f}", fontsize=14)

    # COLUMN 2: High Exposure
    add_basemap_to_ax(ax_high)
    city_prob_pixels.plot(column=f'Prob_High_{y}', ax=ax_high, cmap='YlOrRd', vmin=0, vmax=vmax_high_prob, alpha=0.8, zorder=2)
    ax_high.set_title(f"{y} High Susceptibility\nAccumulated: {high_val:,.0f} ({pct_val:.1f}%)", fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig3.savefig(OUTPUT_PATH / f"{state_code}_{city_name}_Prob_Pixel_Grid.png", dpi=300, bbox_inches='tight')
plt.close(fig3)

print(f"\nAll plots saved to: {OUTPUT_PATH}")