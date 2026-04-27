# %% [markdown]
# # HISDAC 'NobuiltYear' Gap Analyzer
# #### Daniel Acosta-Reyes
# %%
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# --- 1. Setup Paths ---
# ==============================================================================
BASE_DIR = Path("/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics")
DATA_PATH = BASE_DIR / "Data"
ANALYSIS_PATH = BASE_DIR / "Analysis"

NOBUILT_TIF = DATA_PATH / "HISDAC_US_V2/Historical_Settlement_Year_Built_Layer_1810-2020_V2/NobuiltYear.tif"
SHAPEFILE_PATH = DATA_PATH / "tl_2024_us_state" / "tl_2024_us_state.shp"

# ==============================================================================
# --- 2. User Input & Geometry Extraction ---
# ==============================================================================
state_code = input("Enter 2-letter State Code to test (e.g., IL, WA): ").strip().upper()

print(f"\n1) Loading geometries for {state_code}...")
us_states = gpd.read_file(SHAPEFILE_PATH)
state_boundary = us_states[us_states['STUSPS'] == state_code]

if state_boundary.empty:
    print(f"[ERROR] Could not find state {state_code} in shapefile.")
    exit()

# ==============================================================================
# --- 3. Raster Clipping & Quantification ---
# ==============================================================================
print(f"2) Clipping NobuiltYear.tif to {state_code} (This may take a moment)...")

with rasterio.open(NOBUILT_TIF) as src:
    # Project boundary to raster CRS and apply the 1000m buffer for coastal precision
    state_geom_proj = state_boundary.to_crs(src.crs)
    safe_boundary = state_geom_proj.geometry.buffer(1000)
    
    # Clip raster
    data, transform = mask(src, safe_boundary, crop=True, all_touched=True)
    band = data[0].astype(np.float32)
    
    # Handle NoData
    if src.nodata is not None:
        band[band == src.nodata] = np.nan
        
    # Sum all structures missing year-built data
    nobuilt_total = np.nansum(band)

# ==============================================================================
# --- 4. Cross-Reference with Analytics ---
# ==============================================================================
print("3) Cross-referencing with State Analytics...")

csv_path = ANALYSIS_PATH / "Production_analytics" / state_code / f"{state_code}_Metrics_Summary.csv"

if csv_path.exists():
    df = pd.read_csv(csv_path)
    
    # Extract existing metrics
    hisdac_val = float(df[df['Metric'] == 'HISDAC Total 2020']['Value'].values[0])
    led_val = float(df[df['Metric'] == 'LED Total 2020']['Value'].values[0])
    
    # Calculate the missing gap (absolute number of buildings HISDAC is short)
    missing_gap = max(0, led_val - hisdac_val) 
    
    print("\n" + "="*50)
    print(f"📊 NOBUILT-YEAR GAP ANALYSIS: {state_code} 📊")
    print("="*50)
    print(f"Total Modern LED Buildings:       {led_val:,.0f}")
    print(f"HISDAC Total (with Known Years):  {hisdac_val:,.0f}")
    print(f"--> The Missing Building Gap:     {missing_gap:,.0f}\n")
    
    print(f"Structures in NobuiltYear.tif:    {nobuilt_total:,.0f}")
    
    if missing_gap > 0:
        pct_explained = (nobuilt_total / missing_gap) * 100
        print(f"--> NobuiltYear explains {pct_explained:.1f}% of the missing HISDAC gap!")
    else:
        print("--> HISDAC already exceeds LED in this state (Survivorship bias).")
        print(f"    Adding NobuiltYear pushes the HISDAC total to: {(hisdac_val + nobuilt_total):,.0f}")
    print("="*50)
else:
    print(f"\n[INFO] Analytics CSV not found for {state_code}. Here is the raw Nobuilt sum:")
    print(f"Structures in NobuiltYear.tif: {nobuilt_total:,.0f}")

# %% [markdown]
# # HISDAC 'NobuiltYear' Gap Analyzer - National Batch Run
# #### Daniel Acosta-Reyes
# %%
# %% [markdown]
# # Proof 1: Volume Closure Test (National 5% Validation)
# #### Daniel Acosta-Reyes
# %%
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# --- 1. Setup Paths & State List ---
# ==============================================================================
print("1) Initializing National Volume Closure Test...")

BASE_DIR = Path("/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics")
DATA_PATH = BASE_DIR / "Data"
ANALYSIS_PATH = BASE_DIR / "Analysis"

NOBUILT_TIF = DATA_PATH / "HISDAC_US_V2/Historical_Settlement_Year_Built_Layer_1810-2020_V2/NobuiltYear.tif"
SHAPEFILE_PATH = DATA_PATH / "tl_2024_us_state" / "tl_2024_us_state.shp"

# Output CSV saved directly to the requested base directory
OUTPUT_CSV = BASE_DIR / "National_Volume_Closure_Test.csv"

STATES_LOWER_48 = [
    'AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'ID', 'IL', 'IN', 'IA', 
    'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 
    'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 
    'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
]

# ==============================================================================
# --- 2. Load Base Geometries ---
# ==============================================================================
print("2) Loading National Geometries...")
us_states = gpd.read_file(SHAPEFILE_PATH)

results_data = []
skipped_states = []

# ==============================================================================
# --- 3. Master Loop: Raster Extraction & Error Calculation ---
# ==============================================================================
print("3) Scanning HISDAC Nobuilt Raster across 48 states...\n")

# Open the massive TIF exactly once for memory efficiency
with rasterio.open(NOBUILT_TIF) as src:
    raster_crs = src.crs
    
    for state_code in STATES_LOWER_48:
        # 1. Check if analytics exist for this state first to avoid useless processing
        csv_path = ANALYSIS_PATH / "Production_analytics" / state_code / f"{state_code}_Metrics_Summary.csv"
        
        if not csv_path.exists():
            skipped_states.append(state_code)
            continue
            
        print(f"   -> Processing {state_code}...")
        
        # 2. Extract State Geometry and Buffer
        state_boundary = us_states[us_states['STUSPS'] == state_code]
        state_geom_proj = state_boundary.to_crs(raster_crs)
        safe_boundary = state_geom_proj.geometry.buffer(1000)
        
        # 3. Clip the Raster
        data, transform = mask(src, safe_boundary, crop=True, all_touched=True)
        band = data[0].astype(np.float32)
        
        if src.nodata is not None:
            band[band == src.nodata] = np.nan
            
        nobuilt_total = np.nansum(band)
        
        # 4. Extract Existing Metrics
        df_metrics = pd.read_csv(csv_path)
        hisdac_known = float(df_metrics[df_metrics['Metric'] == 'HISDAC Total 2020']['Value'].values[0])
        led_total = float(df_metrics[df_metrics['Metric'] == 'LED Total 2020']['Value'].values[0])
        
        # 5. Calculate Closure Mathematics
        hisdac_combined = hisdac_known + nobuilt_total
        missing_gap_initial = max(0, led_total - hisdac_known)
        
        # Absolute Percentage Error (APE): |(LED - Combined HISDAC) / LED| * 100
        if led_total > 0:
            abs_pct_error = (abs(led_total - hisdac_combined) / led_total) * 100
            raw_pct_diff = ((hisdac_combined - led_total) / led_total) * 100 # Positive = HISDAC overestimates, Negative = HISDAC underestimates
        else:
            abs_pct_error = np.nan
            raw_pct_diff = np.nan
            
        results_data.append({
            'State': state_code,
            'LED_Modern_Total': led_total,
            'HISDAC_Known_Years': hisdac_known,
            'HISDAC_Nobuilt_Years': nobuilt_total,
            'HISDAC_Combined_Total': hisdac_combined,
            'Initial_Missing_Gap': missing_gap_initial,
            'Raw_Percentage_Diff': raw_pct_diff,
            'Absolute_Percentage_Error': abs_pct_error
        })

# ==============================================================================
# --- 4. National Summary & Export ---
# ==============================================================================
if not results_data:
    print("\n[ERROR] No valid state analytics found. Run the main pipeline first.")
    exit()

results_df = pd.DataFrame(results_data)

# Calculate true national aggregate error
nat_led = results_df['LED_Modern_Total'].sum()
nat_hisdac_combined = results_df['HISDAC_Combined_Total'].sum()
national_ape = (abs(nat_led - nat_hisdac_combined) / nat_led) * 100

results_df.to_csv(OUTPUT_CSV, index=False)

print("\n" + "="*60)
print("🇺🇸 NATIONAL VOLUME CLOSURE TEST SUMMARY 🇺🇸")
print("="*60)
print(f"Total States Processed:     {len(results_df)}")
if skipped_states:
    print(f"States Skipped (No Data):   {len(skipped_states)}")
print("-" * 60)
print(f"National LED Total:         {nat_led:,.0f} buildings")
print(f"National HISDAC Combined:   {nat_hisdac_combined:,.0f} buildings")
print(f"--> NATIONAL ABSOLUTE ERROR: {national_ape:.2f}%")
print("="*60)
print(f"Full dataset exported to: {OUTPUT_CSV.name}")