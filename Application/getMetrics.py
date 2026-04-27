# %% [markdown]
# # National Metrics Aggregation & Mapping
# #### Daniel Acosta-Reyes
# %%
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# --- 1. Setup Paths & State List ---
# ==============================================================================
print("1) Setting up paths and state lists...")

BASE_DIR = Path("/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics")
ANALYTICS_DIR = BASE_DIR / "Analysis" / "Production_analytics"
SHAPEFILE_PATH = BASE_DIR / "Data" / "tl_2024_us_state" / "tl_2024_us_state.shp"
OUTPUT_DIR = BASE_DIR / "Analysis" / "National_Summary"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Full list of lower 48 states + DC
STATES_LOWER_48 = [
    'AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'ID', 'IL', 'IN', 'IA', 
    'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 
    'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 
    'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
]

# ==============================================================================
# --- 2. Iterate and Extract Metrics ---
# ==============================================================================
print("2) Scraping metrics from state CSV files...")

metrics_data = []
missing_states = []

for state in STATES_LOWER_48:
    csv_path = ANALYTICS_DIR / state / f"{state}_Metrics_Summary.csv"
    
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        
        # Helper function to extract specific values safely and round to 3 decimals
        def get_val(metric_name, col_name='Value'):
            match = df[df['Metric'] == metric_name]
            if not match.empty:
                val = match[col_name].values[0]
                if pd.notnull(val):
                    return round(float(val), 3)
            return None

        # Extract core volumes and differences (rounded to 3 decimals)
        hisdac_tot = get_val('HISDAC Total 2020', 'Value')
        led_tot = get_val('LED Total 2020', 'Value')
        abs_diff = get_val('Absolute Difference', 'Value')
        pct_diff = get_val('% Difference', 'Value')
        
        # Extract R^2 for all three methods
        r2_raw = get_val('R^2', 'Raw_vs_HISDAC')
        r2_exp = get_val('R^2', 'Expected_vs_HISDAC')
        r2_map = get_val('R^2', 'MAP_vs_HISDAC')

        # Extract Pearson_r for all three methods
        pearson_raw = get_val('Pearson_r', 'Raw_vs_HISDAC')
        pearson_exp = get_val('Pearson_r', 'Expected_vs_HISDAC')
        pearson_map = get_val('Pearson_r', 'MAP_vs_HISDAC')
        
        metrics_data.append({
            'STUSPS': state,
            'HISDAC_Total_2020': hisdac_tot,
            'LED_Total_2020': led_tot,
            'Absolute_Difference': abs_diff,
            'Percent_Difference': pct_diff,
            'R2_Raw': r2_raw,
            'Pearson_r_Raw': pearson_raw,
            'R2_Expected': r2_exp,
            'Pearson_r_Expected': pearson_exp,
            'R2_MAP': r2_map,
            'Pearson_r_MAP': pearson_map
        })
    else:
        missing_states.append(state)

national_df = pd.DataFrame(metrics_data)
national_df.to_csv(OUTPUT_DIR / "National_Metrics_Summary.csv", index=False)

print(f"   -> Successfully processed {len(national_df)} states.")
if missing_states:
    print(f"   -> Missing data for {len(missing_states)} states: {', '.join(missing_states)}")

if national_df.empty:
    print("[ERROR] No state metrics files found. Exiting.")
    exit()

# ==============================================================================
# --- 3. Spatial Join to TIGER Shapefile ---
# ==============================================================================
print("3) Joining metrics to National Shapefile...")

us_states = gpd.read_file(SHAPEFILE_PATH)

# Filter strictly to the lower 48 states
us_lower_48 = us_states[us_states['STUSPS'].isin(STATES_LOWER_48)].copy()

# Reproject to Albers Equal Area (EPSG:5070) for visually accurate US continental maps
us_lower_48 = us_lower_48.to_crs("EPSG:5070")

# Merge the metrics
map_gdf = us_lower_48.merge(national_df, on='STUSPS', how='left')

# ==============================================================================
# --- 4. Plotting the National Map ---
# ==============================================================================
print("4) Generating National Diverging Map...")

fig, ax = plt.subplots(1, 1, figsize=(16, 10))

# Create a TwoSlopeNorm to explicitly center the colormap at 0.0
# This ensures 0 is white/neutral, negative is red, positive is blue (or vice versa)
vmax = map_gdf['Percent_Difference'].max()
vmin = map_gdf['Percent_Difference'].min()

# Handle edge case where data is one-sided
if pd.isna(vmin) or pd.isna(vmax):
    print("Not enough data to plot the map. Need at least one processed state.")
else:
    # Ensure the colorbar is balanced even if the data is skewed
    abs_max = max(abs(vmin), abs(vmax), 1.0) # Fallback to 1.0 if both are 0
    
    norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    # Plot base states (light gray for states that haven't been processed yet)
    us_lower_48.plot(ax=ax, color='#e0e0e0', edgecolor='white', linewidth=0.5)

    # Plot states with data
    map_gdf.dropna(subset=['Percent_Difference']).plot(
        column='Percent_Difference',
        cmap='RdBu', # Red (negative) to Blue (positive)
        norm=norm,
        linewidth=0.8,
        ax=ax,
        edgecolor='black',
        legend=True,
        legend_kwds={
            'label': "Difference in Volume (%)\n(LED vs. HISDAC)",
            'orientation': "horizontal",
            'shrink': 0.6,
            'pad': 0.02
        }
    )

    ax.set_title("National Concordance Validation: LED Modern Inventory vs. HISDAC Historical Baseline\nPercentage Volume Discrepancy (2020)", 
                 fontsize=18, fontweight='bold')
    ax.set_axis_off()

    plt.tight_layout()
    output_map_path = OUTPUT_DIR / "National_Percent_Difference_Map.png"
    fig.savefig(output_map_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"=== COMPLETE: Map and CSV saved to {OUTPUT_DIR} ===")