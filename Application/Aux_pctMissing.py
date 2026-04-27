# %% [markdown]
# # The HISDAC Blind Spot: National Quantification
# #### Daniel Acosta-Reyes
# %%
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# --- 1. Setup Paths ---
# ==============================================================================
print("1) Initializing HISDAC Blind Spot Analysis...")

DATA_PATH = Path("/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics/Data")
ANALYSIS_PATH = Path("/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics/Analysis")
NATIONAL_OUT = ANALYSIS_PATH / "National_Summary"
NATIONAL_OUT.mkdir(parents=True, exist_ok=True)

STATES_LOWER_48 = [
    'AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'ID', 'IL', 'IN', 'IA', 
    'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 
    'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 
    'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
]

YEARS = np.arange(1920, 2025, 5)

# ==============================================================================
# --- 2. Master Loop: Calculate Blind Spots ---
# ==============================================================================
blind_spot_data = []

print("2) Scanning State Master Matrices...")

for STATE_CODE in STATES_LOWER_48:
    PROD_PATH = DATA_PATH / "Production_set" / STATE_CODE
    master_file = PROD_PATH / f"{STATE_CODE}_Master_Spatiotemporal_Matrix.parquet"
    
    if not master_file.exists():
        continue
        
    # Load Master Matrix
    df = pd.read_parquet(master_file)
    
    # Calculate Totals per Pixel
    df['Total_HISDAC'] = df['C_BUPL2020'].fillna(0)
    
    led_cols = [f'LED_{y}' for y in YEARS if f'LED_{y}' in df.columns]
    df['Total_LED'] = df[led_cols].fillna(0).sum(axis=1)
    
    # --- PIXEL METRICS ---
    # How many pixels have at least one modern building?
    pixels_with_led = (df['Total_LED'] > 0).sum()
    
    # How many of those pixels have ZERO historical HISDAC records?
    pixels_blind = ((df['Total_LED'] > 0) & (df['Total_HISDAC'] == 0)).sum()
    
    # --- BUILDING METRICS ---
    # How many total modern buildings exist in this state?
    buildings_total = df['Total_LED'].sum()
    
    # How many of those buildings are sitting in the blind pixels?
    buildings_in_blind_pixels = df.loc[(df['Total_LED'] > 0) & (df['Total_HISDAC'] == 0), 'Total_LED'].sum()

    blind_spot_data.append({
        'State': STATE_CODE,
        'Pixels_with_LED': pixels_with_led,
        'Pixels_HISDAC_Blind': pixels_blind,
        'Buildings_Total': buildings_total,
        'Buildings_in_Blind_Pixels': buildings_in_blind_pixels
    })

# ==============================================================================
# --- 3. National Aggregation & Export ---
# ==============================================================================
results_df = pd.DataFrame(blind_spot_data)

# Calculate Percentages per State
results_df['Pct_Pixels_Blind'] = (results_df['Pixels_HISDAC_Blind'] / results_df['Pixels_with_LED']) * 100
results_df['Pct_Buildings_Blind'] = (results_df['Buildings_in_Blind_Pixels'] / results_df['Buildings_Total']) * 100

# Calculate True National Totals
nat_pixels_led = results_df['Pixels_with_LED'].sum()
nat_pixels_blind = results_df['Pixels_HISDAC_Blind'].sum()
nat_bldgs_tot = results_df['Buildings_Total'].sum()
nat_bldgs_blind = results_df['Buildings_in_Blind_Pixels'].sum()

nat_pct_pixels = (nat_pixels_blind / nat_pixels_led) * 100
nat_pct_bldgs = (nat_bldgs_blind / nat_bldgs_tot) * 100

# Save to CSV
results_df.to_csv(NATIONAL_OUT / "HISDAC_Blind_Spot_Analysis.csv", index=False)

# Print the National Summary to the console
print("\n" + "="*50)
print("NATIONAL HISDAC BLIND SPOT SUMMARY")
print("="*50)
print(f"Total Pixels with Modern Buildings:   {nat_pixels_led:,.0f}")
print(f"Pixels with 0 Historical Records:     {nat_pixels_blind:,.0f}")
print(f"--> PERCENTAGE OF PIXELS BLIND:       {nat_pct_pixels:.2f}%\n")

print(f"Total Modern Buildings (LED):         {nat_bldgs_tot:,.0f}")
print(f"Buildings sitting in Blind Pixels:    {nat_bldgs_blind:,.0f}")
print(f"--> PERCENTAGE OF BUILDINGS BLIND:    {nat_pct_bldgs:.2f}%")
print("="*50)
print(f"Data saved to {NATIONAL_OUT / 'HISDAC_Blind_Spot_Analysis.csv'}")

# %% [markdown]
# # HISDAC Blind Spot Spatial Extraction & Mapping
# #### Daniel Acosta-Reyes
# %%
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# --- 1. Setup Paths ---
# ==============================================================================
print("1) Initializing Blind Spot Spatial Mapper...")

BASE_DIR = Path("/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics")
DATA_PATH = BASE_DIR / "Data"
ANALYSIS_PATH = BASE_DIR / "Analysis"
SHAPEFILE_PATH = DATA_PATH / "tl_2024_us_state" / "tl_2024_us_state.shp"

STATES_LOWER_48 = [
    'AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'ID', 'IL', 'IN', 'IA', 
    'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 
    'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 
    'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
]

YEARS = np.arange(1920, 2025, 5)

# Load State Boundaries once for the basemaps
us_states = gpd.read_file(SHAPEFILE_PATH)

# ==============================================================================
# --- 2. Master Loop: Extract, Save GPKG, and Plot ---
# ==============================================================================
for STATE_CODE in STATES_LOWER_48:
    PROD_PATH = DATA_PATH / "Production_set" / STATE_CODE
    ANALYTICS_PATH = ANALYSIS_PATH / "Production_analytics" / STATE_CODE
    
    master_file = PROD_PATH / f"{STATE_CODE}_Master_Spatiotemporal_Matrix.parquet"
    anchor_file = PROD_PATH / f"{STATE_CODE}_HISDAC_Spatial_Anchor.gpkg"
    
    if not master_file.exists() or not anchor_file.exists():
        continue
        
    print(f"\nProcessing Blind Spots for {STATE_CODE}...")
    ANALYTICS_PATH.mkdir(parents=True, exist_ok=True)
    
    # --- A. Load and Filter Data ---
    df = pd.read_parquet(master_file)
    
    df['Total_HISDAC'] = df['C_BUPL2020'].fillna(0)
    led_cols = [f'LED_{y}' for y in YEARS if f'LED_{y}' in df.columns]
    df['Total_LED'] = df[led_cols].fillna(0).sum(axis=1)
    
    # FILTER: Keep ONLY pixels where LED exists but HISDAC is completely zero
    blind_df = df[(df['Total_LED'] > 0) & (df['Total_HISDAC'] == 0)].copy()
    
    if blind_df.empty:
        print(f"   -> No blind spots found in {STATE_CODE}.")
        continue
        
    print(f"   -> Found {len(blind_df):,.0f} blind pixels. Merging geometries...")
    
    # --- B. Spatial Merge ---
    # We only read the columns we need to save memory
    spatial_anchor = gpd.read_file(anchor_file, columns=['HISDAC_id', 'geometry'])
    
    blind_gdf = spatial_anchor.merge(blind_df[['HISDAC_id', 'Total_LED']], on='HISDAC_id', how='inner')
    
    # --- C. Export the GIS Layer ---
    out_gpkg = PROD_PATH / f"{STATE_CODE}_HISDAC_Blind_Spots.gpkg"
    blind_gdf.to_file(out_gpkg, driver="GPKG")
    print(f"   -> Saved spatial grid to {out_gpkg.name}")

    # --- D. Generate the Map ---
    print(f"   -> Rendering Map...")
    
    state_boundary = us_states[us_states['STUSPS'] == STATE_CODE]
    if not state_boundary.empty:
        # Reproject boundary to match the pixels (Albers/Meter based)
        state_boundary = state_boundary.to_crs(blind_gdf.crs)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot State Boundary as background
    if not state_boundary.empty:
        state_boundary.plot(ax=ax, color='#f4f6f6', edgecolor='#2c3e50', linewidth=1.5)
        
    # Plot the Blind Pixels, colored by how many modern buildings are hiding in them
    # We use a logarithmic normalization because some pixels might have 1 building, others might have 500
    vmax = blind_gdf['Total_LED'].quantile(0.99) # Cap at 99th percentile to prevent a few outliers from washing out the colors
    vmin = 1
    
    plot = blind_gdf.plot(
        column='Total_LED', 
        ax=ax, 
        cmap='YlOrRd', 
        markersize=2, # If they look too small on large states, Geopandas handles polygon sizing natively, so markersize is a fallback if centroids were used. Since these are polygons, it uses the polygon area.
        linewidth=0,
        legend=True,
        norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
        legend_kwds={
            'label': "Number of Missing Modern Buildings per 250m Pixel",
            'orientation': "horizontal",
            'shrink': 0.6,
            'pad': 0.05
        }
    )
    
    ax.set_title(f"The 'HISDAC Blind Spot': {STATE_CODE}\nModern Development with Zero Historical Tax Records", fontsize=16, fontweight='bold')
    ax.set_axis_off()
    
    # Calculate quick stats for the subtitle
    total_blind_bldgs = blind_gdf['Total_LED'].sum()
    ax.text(0.5, -0.1, f"Total Invisible Buildings: {total_blind_bldgs:,.0f} | Total Blind Pixels: {len(blind_gdf):,.0f}", 
            transform=ax.transAxes, ha='center', fontsize=12, style='italic', color='#34495e')

    plt.tight_layout()
    map_out = ANALYTICS_PATH / f"{STATE_CODE}_Blind_Spot_Map.png"
    fig.savefig(map_out, dpi=300, bbox_inches='tight')
    plt.close(fig)

print("\n=== BLIND SPOT MAPPING COMPLETE ===")
# %%
