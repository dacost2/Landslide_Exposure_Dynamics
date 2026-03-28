# %% [markdown]
# Test LED against NHGIS data
# Daniel Acosta-Reyes
# %%
from __future__ import annotations

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

led_inventory = gpd.read_file('/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics/Data/Testing_set/WA_2020_LED_with_NHGIS_id.gpkg')
# create subset of data by 10-year interval using 'med_yr_blt' column
led_inventory['GISJOIN'] = led_inventory['GISJOIN'].astype(str).str.strip().str.upper()
led_inventory['med_yr_blt'] = pd.to_numeric(led_inventory['med_yr_blt'], errors='coerce')
led_inventory['decade'] = ((led_inventory['med_yr_blt'] + 9) // 10) * 10
led_inventory_all = led_inventory.copy()
led_decades_all = sorted(pd.Series(led_inventory_all['decade'].dropna()).astype(int).unique().tolist())
valid_decades = [1970, 1980, 1990, 2000, 2010, 2020]
led_inventory = led_inventory[led_inventory['decade'].isin(valid_decades)]
# Create individual GeoDataFrames for each decade
led_decades = {decade: led_inventory[led_inventory['decade'] == decade] for decade in led_inventory['decade'].unique()}

# Load NHGIS data
nhgis_data = pd.read_csv('/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics/Data/NHGIS/nhgis0001_csv/nhgis0001_ts_nominal_tract.csv', encoding='iso-8859-1')
nhgis_col = ['NHGISCODE', 'STATE', 'A41AA1970', 'A41AA1980', 'A41AA1990', 'A41AA2000', 'A41AA2010', 'A41AA2020']
nhgis_data = nhgis_data[nhgis_col]
nhgis_data = nhgis_data[nhgis_data['STATE'] == 'Washington'].copy()
nhgis_data = nhgis_data.rename(columns={'NHGISCODE': 'GISJOIN'})
nhgis_data['GISJOIN'] = nhgis_data['GISJOIN'].astype(str).str.strip().str.upper()
for col in ['A41AA1970', 'A41AA1980', 'A41AA1990', 'A41AA2000', 'A41AA2010', 'A41AA2020']:
    nhgis_data[col] = pd.to_numeric(nhgis_data[col], errors='coerce')
print(f"NHGIS records after STATE filter (Washington): {len(nhgis_data)}")
# %%
# Aggregate LED counts by GISJOIN and decade (from earliest LED decade), then merge into NHGIS big table.
led_counts_long = (
    led_inventory_all.dropna(subset=['decade'])
    .assign(decade=lambda df: df['decade'].astype(int))
    .groupby(['GISJOIN', 'decade'])
    .size()
    .reset_index(name='led_count')
)
led_counts_wide = led_counts_long.pivot(index='GISJOIN', columns='decade', values='led_count').fillna(0)
led_counts_wide.columns = [f'led_{int(c)}' for c in led_counts_wide.columns]
led_counts_wide = led_counts_wide.reset_index()

nhgis_led = nhgis_data.merge(led_counts_wide, on='GISJOIN', how='left')
for decade in led_decades_all:
    led_col = f'led_{decade}'
    if led_col not in nhgis_led.columns:
        nhgis_led[led_col] = 0
    nhgis_led[led_col] = pd.to_numeric(nhgis_led[led_col], errors='coerce').fillna(0)

# Add cumulative LED columns to match NHGIS cumulative interpretation.
cumulative_running = np.zeros(len(nhgis_led), dtype=float)
for decade in led_decades_all:
    led_col = f'led_{decade}'
    acc_col = f'led_{decade}_acc'
    cumulative_running = cumulative_running + nhgis_led[led_col].to_numpy(dtype=float)
    nhgis_led[acc_col] = cumulative_running

print(
    f'Added cumulative LED columns from earliest decade {min(led_decades_all)} to {max(led_decades_all)}'
)

# Keep decade-specific tables for plotting.
led_decades = {}
for decade in valid_decades:
    led_col = f'led_{decade}'
    nhgis_col = f'A41AA{decade}'
    led_decades[str(decade)] = nhgis_led[['GISJOIN', nhgis_col, led_col]].rename(columns={led_col: 'led_count'})

print('Data loaded and merged successfully. Ready for analysis.')
# %%

# Create one grid plot (2x3) for all decades
decades_to_plot = ['1970', '1980', '1990', '2000', '2010', '2020']
fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
axes = axes.flatten()

for i, decade in enumerate(decades_to_plot):
    ax = axes[i]
    plot_df = led_decades[decade].copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.dropna(subset=[f'A41AA{decade}', 'led_count'])
    print(f"{decade}: plotting {len(plot_df)} points")

    if plot_df.empty:
        ax.set_title(f"{decade}s (no valid data)")
        ax.grid(True, which="both", ls="--", linewidth=0.5)
        continue

    ax.scatter(plot_df[f'A41AA{decade}'], plot_df['led_count'], alpha=0.5, s=10)
    ax.set_title(f'{decade}s')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", ls="--", linewidth=0.5)

for ax in axes:
    ax.set_xlabel('NHGIS Built-up Count')
    ax.set_ylabel('LED Count')

fig.suptitle('LED Count vs NHGIS Built-up Count by Decade', fontsize=14)
plt.tight_layout()
plt.show()
# %%
# Total count comparison by decade (LED vs NHGIS) in one side-by-side bar plot.
led_total_by_decade = (
    led_inventory_all.dropna(subset=['decade'])
    .groupby('decade')
    .size()
    .rename('led_total')
)

nhgis_total_by_decade = pd.Series(
    {
        1970: nhgis_data['A41AA1970'].sum(skipna=True),
        1980: nhgis_data['A41AA1980'].sum(skipna=True),
        1990: nhgis_data['A41AA1990'].sum(skipna=True),
        2000: nhgis_data['A41AA2000'].sum(skipna=True),
        2010: nhgis_data['A41AA2010'].sum(skipna=True),
        2020: nhgis_data['A41AA2020'].sum(skipna=True),
    },
    name='nhgis_total',
)

all_decades = sorted(set(led_total_by_decade.index.astype(int)).union(set(nhgis_total_by_decade.index.astype(int))))
total_df = pd.DataFrame({'decade': all_decades})
total_df['led_total'] = total_df['decade'].map(led_total_by_decade).fillna(0)
total_df['nhgis_total'] = total_df['decade'].map(nhgis_total_by_decade).fillna(0)

x = np.arange(len(total_df))
width = 0.42

plt.figure(figsize=(14, 6))
plt.bar(x - width / 2, total_df['led_total'], width=width, label='LED total')
plt.bar(x + width / 2, total_df['nhgis_total'], width=width, label='NHGIS total')
plt.xticks(x, total_df['decade'].astype(int), rotation=45)
plt.xlabel('Decade')
plt.ylabel('Total building count')
plt.title('Total Building Count by Decade: LED vs NHGIS')
plt.grid(True, axis='y', ls='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# Cumulative total comparison by decade (LED cumulative vs NHGIS cumulative).
total_df = total_df.sort_values('decade').reset_index(drop=True)
total_df['led_total_acc'] = total_df['led_total'].cumsum()
total_df['nhgis_total_acc'] = total_df['nhgis_total']

x_acc = np.arange(len(total_df))

plt.figure(figsize=(14, 6))
plt.bar(x_acc - width / 2, total_df['led_total_acc'], width=width, label='LED cumulative total')
plt.bar(x_acc + width / 2, total_df['nhgis_total_acc'], width=width, label='NHGIS cumulative total')
plt.xticks(x_acc, total_df['decade'].astype(int), rotation=45)
plt.xlabel('Decade')
plt.ylabel('Cumulative building count')
plt.title('Cumulative Building Count by Decade: LED vs NHGIS')
plt.grid(True, axis='y', ls='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()
# %%
# Simple one-GISJOIN plot using the enriched master table.
def plot_gisjoin_from_enriched_table(
    nhgis_led_df: pd.DataFrame,
    led_points_df: pd.DataFrame,
    gisjoin_id: str = 'G5300330009701',
    debug: bool = True,
) -> pd.DataFrame:
    """Select one GISJOIN and plot cumulative LED counts (plus NHGIS cumulative where available)."""
    gisjoin_id = gisjoin_id.strip().upper()

    # Normalize inside the function so it remains correct even if stale kernel variables were created before edits.
    nhgis_led_local = nhgis_led_df.copy()
    led_points_local = led_points_df.copy()
    nhgis_led_local['GISJOIN'] = nhgis_led_local['GISJOIN'].astype(str).str.strip().str.upper()
    led_points_local['GISJOIN'] = led_points_local['GISJOIN'].astype(str).str.strip().str.upper()

    row_df = nhgis_led_local[nhgis_led_local['GISJOIN'] == gisjoin_id]
    raw_points = led_points_local[led_points_local['GISJOIN'] == gisjoin_id]

    if row_df.empty:
        raise ValueError(f'GISJOIN {gisjoin_id} not found in master table.')
    row = row_df.iloc[0]

    led_cols = [
        c for c in nhgis_led_local.columns
        if c.startswith('led_') and not c.endswith('_acc') and c.split('_')[1].isdigit()
    ]
    if not led_cols:
        raise ValueError('No LED decade columns found (expected led_XXXX).')

    years = sorted([int(c.split('_')[1]) for c in led_cols])
    ts_df = pd.DataFrame({'year': years})

    # Recompute cumulative series from decade counts so plotting is always consistent.
    ts_df['led_count'] = [pd.to_numeric(row.get(f'led_{y}', 0), errors='coerce') for y in years]
    ts_df['led_count'] = ts_df['led_count'].fillna(0)
    ts_df['led_count_acc'] = ts_df['led_count'].cumsum()
    ts_df['nhgis_count'] = [row[f'A41AA{y}'] if f'A41AA{y}' in nhgis_led_local.columns else np.nan for y in years]

    if debug:
        raw_decade_counts = pd.Series(dtype=float)
        if not raw_points.empty:
            raw_decade_counts = (
                raw_points['decade']
                .dropna()
                .astype(int)
                .value_counts()
                .sort_index()
            )

        merged_nonzero = ts_df.loc[ts_df['led_count'] > 0, ['year', 'led_count']]
        print(f'GISJOIN selected: {gisjoin_id}')
        print(f'Raw LED points in this GISJOIN: {len(raw_points)}')
        if not raw_decade_counts.empty:
            print('Raw LED decade counts (point inventory):')
            print(raw_decade_counts)
        else:
            print('Raw LED decade counts (point inventory): none')

        print('Merged LED decade counts (non-zero only):')
        if merged_nonzero.empty:
            print('none')
        else:
            print(merged_nonzero.to_string(index=False))

        raw_total = float(raw_decade_counts.sum()) if not raw_decade_counts.empty else 0.0
        merged_total = float(ts_df['led_count'].sum())
        print(f'Raw total points: {raw_total:.0f}')
        print(f'Merged total from led_XXXX columns: {merged_total:.0f}')
        if raw_total != merged_total:
            print('WARNING: raw vs merged totals differ for this GISJOIN. Re-run all cells from top to rebuild nhgis_led.')

    print(ts_df.head(10))

    plt.figure(figsize=(10, 6))
    plt.plot(ts_df['year'], ts_df['led_count_acc'], marker='o', label='LED Count (Cumulative)')
    plt.plot(ts_df['year'], ts_df['nhgis_count'], marker='o', label='NHGIS Built-up Count (Cumulative)')
    plt.title(f'WA STATE\nCumulative LED vs NHGIS for GISJOIN {gisjoin_id}')
    plt.xlabel('Year')
    plt.ylabel('Cumulative count')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend()
    plt.xlim(min(years), max(years))
    plt.ylim(bottom=0)
    plt.show()

    return ts_df


selected_gisjoin = input('Enter GISJOIN (press Enter for default G5300330009701): ').strip()
if selected_gisjoin == '':
    selected_gisjoin = 'G5300330009701'

_ts_selected = plot_gisjoin_from_enriched_table(
    nhgis_led,
    led_inventory_all,
    gisjoin_id=selected_gisjoin,
)
# %%
