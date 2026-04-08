# %%
# Take Total LED data and create subset GeoPackages for each state.
from pathlib import Path
import geopandas as gpd
from tqdm import tqdm

LED_PATH = Path('/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 1 - Exposure analysis/AGU Submission/AGU Earth Future - Initial Submission Package/data_release/US_final_population_building_inventory_pointData_with_socioeconomic-vMar12-2026.gpkg')
OUTPUT_DIR = Path('/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics/Data/LED_by_State_GPKG')

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading LED data...")
try:
    led_gdf = gpd.read_file(LED_PATH)
    print(f"  -> Loaded {len(led_gdf):,} total records")
except FileNotFoundError:
    raise FileNotFoundError(f"LED file not found: {LED_PATH}")
except Exception as e:
    raise Exception(f"Error loading LED file: {e}")

if led_gdf.geometry.name not in led_gdf.columns:
    raise ValueError("Active geometry column is missing from the LED dataset.")

if led_gdf.crs is None:
    raise ValueError("LED dataset has no CRS; QGIS may fail to place or read geometry correctly.")

# Validate state column exists
if 'NAME' not in led_gdf.columns:
    raise ValueError(f"Column 'NAME' not found. Available columns: {list(led_gdf.columns)}")

states = led_gdf['NAME'].unique()
print(f"Splitting LED data by {len(states)} states and saving as GeoPackage...")
print()

saved_count = 0
total_records = 0
failed_states = []

for state in tqdm(states, desc="Processing states", unit="state"):
    try:
        state_gdf = led_gdf[led_gdf['NAME'] == state].copy()
        n_records = len(state_gdf)
        total_records += n_records

        # Rebuild explicitly as GeoDataFrame to preserve geometry metadata reliably.
        state_gdf = gpd.GeoDataFrame(
            state_gdf,
            geometry=led_gdf.geometry.name,
            crs=led_gdf.crs,
        )
        
        output_path = OUTPUT_DIR / f"{state}_LED.gpkg"
        layer_name = "led_points"

        # GeoPackage is broadly compatible with desktop GIS tools like QGIS.
        state_gdf.to_file(output_path, layer=layer_name, driver="GPKG")

        # Round-trip check catches files that were written but cannot be read as geospatial.
        _check = gpd.read_file(output_path, layer=layer_name)
        if _check.geometry.name not in _check.columns:
            raise ValueError("Round-trip validation failed: geometry column missing after write.")

        saved_count += 1
    except Exception as e:
        failed_states.append((state, str(e)))
        tqdm.write(f"  ERROR: Failed to save {state}: {e}")

print()
print(f"=== Summary ===")
print(f"States processed: {saved_count}/{len(states)}")
print(f"Total records written: {total_records:,}")
print(f"Output directory: {OUTPUT_DIR}")
if failed_states:
    print(f"Failed states: {len(failed_states)}")
    for state, error in failed_states:
        print(f"  - {state}: {error}")
else:
    print("SUCCESS! All states saved.")

