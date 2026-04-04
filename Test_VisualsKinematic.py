# %% [markdown]
# Phase 7: Kinematic Cartography
# Merging tabular kinematics with NHGIS geometries to map exposure dynamics.

# %%
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# --- 1) Setup Paths ---
DATA_PATH = Path("/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics/Data")
TESTING_SET_PATH = DATA_PATH / "Testing_Set"
STATE_FIPS = "53" # Washington State

# NHGIS Shapefile and your new CSV
NHGIS_COUNTY_PATH = DATA_PATH / "NHGIS/nhgis0005_COUNTY_shape/nhgis0005_shapefile_tl2020_us_county_2020.zip"
KINEMATICS_CSV_PATH = TESTING_SET_PATH / "WA_County_Kinematics.csv"

# --- 2) Load and Prep Data ---
print("1) Loading NHGIS County Shapefile...")
# Read the shapefile directly from the zip
counties_gdf = gpd.read_file(f"zip://{NHGIS_COUNTY_PATH}")

print("2) Loading Kinematics CSV...")
county_metrics = pd.read_csv(KINEMATICS_CSV_PATH)
# Ensure the GEOID is a string padded to 5 digits (e.g., '53033' instead of 53033)
county_metrics['County_GEOID'] = county_metrics['County_GEOID'].astype(str).str.zfill(5)

# --- 3) The Spatial-Tabular Bridge ---
print("3) Filtering to WA and Merging...")

# NHGIS shapefiles usually have a 'STATEFP' or 'STATEFP20' column. 
# Let's dynamically find it so the script doesn't break on different NHGIS versions.
state_col = [col for col in counties_gdf.columns if 'STATEFP' in col.upper()][0]
wa_counties = counties_gdf[counties_gdf[state_col] == STATE_FIPS].copy()

# NHGIS uses 'GISJOIN' (e.g., 'G5300330') but usually also includes standard 'GEOID' (e.g., '53033').
# If GEOID exists, we use it. If not, we extract it from GISJOIN.
geoid_candidates = [col for col in wa_counties.columns if col.upper() in ['GEOID', 'GEOID20']]
if not geoid_candidates:
    raise ValueError("NHGIS county file does not include GEOID/GEOID20 for 5-digit county joins.")

join_col = geoid_candidates[0]
wa_counties[join_col] = wa_counties[join_col].astype(str).str.zfill(5)

# Execute the Merge
wa_mapped = wa_counties.merge(county_metrics, left_on=join_col, right_on='County_GEOID', how='left')

# Build decade windows from available Pulse outputs in the kinematics file.
pulse_cols = [
    col for col in county_metrics.columns
    if col.startswith('Pulse_') and col.split('_')[-1].isdigit()
]
pulse_years = sorted(int(col.split('_')[-1]) for col in pulse_cols)
print(f"Available Pulse years in CSV: {pulse_years}")

if len(pulse_years) < 2:
    raise ValueError("Need at least two Pulse_YYYY columns to build decade windows.")

decade_windows = list(zip(pulse_years[:-1], pulse_years[1:]))
for start, end in decade_windows:
    pulse_col = f'Pulse_{end}'
    decade_col = f'PulseDec_{start}_{end}'
    wa_mapped[decade_col] = pd.to_numeric(wa_mapped[pulse_col], errors='coerce')

fill_cols = ['Total_Exposure', 'Velocity_2020', 'Accel_2020']
fill_cols = [col for col in fill_cols if col in wa_mapped.columns]
wa_mapped[fill_cols] = wa_mapped[fill_cols].fillna(0)

decade_rows = []
cumulative_exposure = 0
for start, end in decade_windows:
    decade_col = f'PulseDec_{start}_{end}'
    decade_series = pd.to_numeric(wa_mapped[decade_col], errors='coerce')
    if decade_series.notna().any():
        added = int(decade_series.sum())
        cumulative_exposure += added
        data_status = 'available'
    else:
        added = pd.NA
        data_status = 'missing'

    decade_rows.append({
        'Decade_Window': f'{start}-{end}',
        'Pulse_Added': added,
        'Cumulative_Exposure': int(cumulative_exposure) if data_status == 'available' else pd.NA,
        'Data_Status': data_status,
    })

decade_summary = pd.DataFrame(decade_rows)
print(f"\nDecade summary ({len(decade_summary)} rows):")
print(decade_summary.to_string(index=False))


def get_time_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    """Return columns like Prefix_YYYY sorted by year."""
    cols = [
        col for col in df.columns
        if col.startswith(prefix) and col.split('_')[-1].isdigit()
    ]
    return sorted(cols, key=lambda c: int(c.split('_')[-1]))


def plot_time_panels(
    gdf: gpd.GeoDataFrame,
    metric_cols: list[str],
    figure_title: str,
    colorbar_label: str,
    cmap: str,
    force_diverging: bool = False,
) -> None:
    """Plot one map per time column with shared color scaling."""
    if not metric_cols:
        print(f"No columns found for: {figure_title}")
        return

    n_panels = len(metric_cols)
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6), dpi=300)
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(figure_title, fontsize=20, fontweight='bold')

    all_values = pd.concat(
        [pd.to_numeric(gdf[col], errors='coerce') for col in metric_cols],
        axis=0,
    )
    min_val = all_values.min(skipna=True)
    max_val = all_values.max(skipna=True)

    use_diverging = force_diverging or (pd.notna(min_val) and min_val < 0)

    if pd.isna(min_val) or pd.isna(max_val):
        min_val, max_val = 0, 1

    if use_diverging:
        max_abs = max(abs(float(min_val)), abs(float(max_val)))
        if max_abs == 0:
            max_abs = 1
        norm = mpl.colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
        vmin = None
        vmax = None
    else:
        norm = None
        vmin = float(min_val)
        vmax = float(max_val)
        if vmin == vmax:
            vmax = vmin + 1

    for ax, col in zip(axes, metric_cols):
        ax.set_axis_off()
        year = col.split('_')[-1]
        gdf.plot(
            column=col,
            cmap=cmap,
            linewidth=0.6,
            ax=ax,
            edgecolor='0.8',
            legend=False,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(year, fontsize=12)

    if use_diverging:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    else:
        sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])
    fig.colorbar(
        sm,
        ax=axes,
        orientation='horizontal',
        fraction=0.03,
        pad=0.03,
        label=colorbar_label,
    )

    plt.tight_layout()
    plt.show()

# --- 4) Cartography ---
# %%
print("4) Generating Maps...")

n_windows = len(decade_windows)
fig, axes = plt.subplots(1, n_windows, figsize=(8 * n_windows, 6), dpi=300)
if n_windows == 1:
    axes = [axes]

first_decade_start = decade_windows[0][0]
last_decade_end = decade_windows[-1][1]
fig.suptitle(
    f"High-Hazard Exposure Pulse by Available Decades: Washington State ({first_decade_start}-{last_decade_end})",
    fontsize=20,
    fontweight='bold',
)

all_decade_values = pd.concat(
    [pd.to_numeric(wa_mapped[f'PulseDec_{start}_{end}'], errors='coerce') for start, end in decade_windows],
    axis=0,
)
vmax = all_decade_values.max(skipna=True)
if pd.isna(vmax) or vmax == 0:
    vmax = 1

# Remove axes for clean maps
for ax in axes:
    ax.set_axis_off()

for ax, (start, end) in zip(axes, decade_windows):
    decade_col = f'PulseDec_{start}_{end}'
    decade_data = pd.to_numeric(wa_mapped[decade_col], errors='coerce')
    if decade_data.notna().any():
        title = f'{start}-{end}'
    else:
        title = f'{start}-{end} (no data)'

    wa_mapped.plot(
        column=decade_col,
        cmap='YlOrRd',
        linewidth=0.6,
        ax=ax,
        edgecolor='0.8',
        legend=False,
        vmin=0,
        vmax=vmax,
    )
    ax.set_title(title, fontsize=12)

norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
sm = mpl.cm.ScalarMappable(norm=norm, cmap='YlOrRd')
sm.set_array([])
fig.colorbar(
    sm,
    ax=axes,
    orientation='horizontal',
    fraction=0.03,
    pad=0.03,
    label='Buildings Added During Decade Window',
)

plt.tight_layout()
plt.show()

velocity_cols = get_time_columns(wa_mapped, 'Velocity_')
print(f"Velocity columns: {velocity_cols}")
plot_time_panels(
    gdf=wa_mapped,
    metric_cols=velocity_cols,
    figure_title='Velocity by Available Years: Washington State',
    colorbar_label='Velocity (Change in Position)',
    cmap='YlGnBu',
)

accel_cols = get_time_columns(wa_mapped, 'Accel_')
print(f"Acceleration columns: {accel_cols}")
plot_time_panels(
    gdf=wa_mapped,
    metric_cols=accel_cols,
    figure_title='Acceleration by Available Years: Washington State',
    colorbar_label='Acceleration (Change in Velocity)',
    cmap='RdBu_r',
    force_diverging=True,
)

print("SUCCESS! Decade maps rendered.")
# %%
