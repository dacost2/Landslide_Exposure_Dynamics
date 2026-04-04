# %%
# %% [markdown]
# Phase 6: Multi-Scale Spatial Aggregation & Kinematics
# Prepares data at Block, Tract, and County levels for mapping and policy evaluation.

# %%
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import requests
from pathlib import Path

# --- 1) Setup & Download (Washington State FIPS: 53) ---
DATA_PATH = Path("/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics/Data")
TESTING_SET_PATH = DATA_PATH / "Testing_Set"
STATE_FIPS = "53" 
YEARS = np.arange(1920, 2025, 5)

print("1) Downloading 2024 Census Boundaries for WA...")
out_folder = DATA_PATH / "census_gov_data" / "tabblock20"
out_folder.mkdir(parents=True, exist_ok=True)

# Download WA Blocks
block_file = f"tl_2024_{STATE_FIPS}_tabblock20.zip"
block_url = f"https://www2.census.gov/geo/tiger/TIGER2024/TABBLOCK20/{block_file}"
block_path = out_folder / block_file

if not block_path.exists():
    print(f"Downloading {block_file}...")
    response = requests.get(block_url, stream=True, timeout=60)
    response.raise_for_status()
    with open(block_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

print("Loading Census Blocks into memory...")
blocks = gpd.read_file(f"zip://{block_path}")
blocks_clean = blocks[['GEOID20', 'geometry']]

# --- 2) The Master Spatial Link ---
print("\n2) Executing Master Spatial Join...")
led_engine = pd.read_parquet(TESTING_SET_PATH / "WA_LED_Monte_Carlo_Engine.parquet")
led_spatial = gpd.read_file(TESTING_SET_PATH / "WA_LED_points-5070.gpkg")

# Merge probability arrays to spatial points
led_inventory = led_spatial.merge(
    led_engine[['prob_distribution']], 
    left_index=True, right_index=True, how='inner'
)

# Ensure CRS match
blocks_clean = blocks_clean.to_crs(led_inventory.crs)

# Join Buildings to Blocks
master_joined = gpd.sjoin(led_inventory, blocks_clean, how="inner", predicate="intersects")
master_joined = master_joined.drop(columns=["index_right"])

# Rename GEOID20 from blocks to avoid ambiguity
master_joined = master_joined.rename(columns={"GEOID20_right": "GEOID20"}) if "GEOID20_right" in master_joined.columns else master_joined

# Generate Hierarchical IDs via String Slicing
# GEOID20 structure: 2-digit state + 3-digit county + 6-digit tract + 4-digit block = 15 digits total
master_joined['County_GEOID'] = master_joined['GEOID20'].str[:5]  # State + County (first 5 digits)
master_joined['Tract_GEOID'] = master_joined['GEOID20'].str[:11]  # State + County + Tract (first 11 digits)

# --- 3) Define Kinematic Functions ---
def calculate_kinematics(group_df):
    """Calculates Position, Velocity, and Acceleration for a specific geographic level."""
    # Convert probability arrays to a 2D matrix
    prob_matrix = np.stack(group_df['prob_distribution'].values)
    
    # 1. Position ($E_t$): Absolute Expected Builds per 5-year interval
    position = np.sum(prob_matrix, axis=0)
    
    # 2. Velocity ($\frac{dE}{dt}$): Change in Position (Current Interval - Previous Interval)
    # Note: Prepend a 0 to maintain array length. Velocity at 1920 is just the 1920 accumulation.
    velocity = np.insert(np.diff(position), 0, position[0])
    
    # 3. Acceleration ($\frac{d^2E}{dt^2}$): Change in Velocity
    acceleration = np.insert(np.diff(velocity), 0, velocity[0])
    
    # Total cumulative exposure up to 2020
    total_exposure = np.sum(position)
    
    return pd.Series({
        'Total_Exposure': total_exposure,
        'Pulse_1990': position[14], # Index 14 is 1990
        'Pulse_2000': position[16],
        'Pulse_2010': position[18],
        'Pulse_2020': position[20],
        'Velocity_2000': velocity[16],
        'Velocity_2020': velocity[20],
        'Accel_2000': acceleration[16],
        'Accel_2020': acceleration[20]
    })

# --- 4) Execute Aggregations ---
print("\n3) Computing Exposure Metrics by Geography...")

# Filter for High Hazard only for this specific policy mapping exercise
high_hazard_df = master_joined[master_joined['susc_class'] == 'high']

print("   -> Computing Block Level (Urban vs Rural)...")
# Group by Block GEOID and UR20
block_metrics = high_hazard_df.groupby(['GEOID20', 'UR20']).apply(calculate_kinematics, include_groups=False).reset_index()

print("   -> Computing Tract Level...")
tract_metrics = high_hazard_df.groupby('Tract_GEOID').apply(calculate_kinematics, include_groups=False).reset_index()

# Calculate Exposure Intensity (Proportional) at Tract Level
# Total buildings in tract (all hazards)
tract_totals = master_joined.groupby('Tract_GEOID').size().reset_index(name='Total_Buildings')
tract_metrics = tract_metrics.merge(tract_totals, on='Tract_GEOID', how='left')
tract_metrics['Intensity_Percent'] = (tract_metrics['Total_Exposure'] / tract_metrics['Total_Buildings']) * 100

print("   -> Computing County Level...")
county_metrics = high_hazard_df.groupby('County_GEOID').apply(calculate_kinematics, include_groups=False).reset_index()

# --- 5) Export for Mapping ---
print("\n4) Exporting datasets for GIS Mapping...")

block_metrics.to_csv(TESTING_SET_PATH / "WA_Block_Kinematics.csv", index=False)
tract_metrics.to_csv(TESTING_SET_PATH / "WA_Tract_Kinematics.csv", index=False)
county_metrics.to_csv(TESTING_SET_PATH / "WA_County_Kinematics.csv", index=False)

print("SUCCESS! Data is ready to be joined to Census shapefiles in QGIS.")

# --- Quick Console Insight ---
# Let's find a county that successfully decelerated in 2020
decelerating_counties = county_metrics[(county_metrics['Velocity_2020'] > 0) & (county_metrics['Accel_2020'] < 0)]
print(f"\nFound {len(decelerating_counties)} WA counties where high-hazard development is growing (Positive Velocity) but slowing down (Negative Acceleration) in 2020.")