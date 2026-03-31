# %% [markdown]
# Compare LED and NHGIS workflows
# Restored script with spatial-index joins, decade aggregation, and cumulative merges.

# %%
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 200)


def normalize_gisjoin(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def decade_from_med_yr_blt(series: pd.Series) -> pd.Series:
    """Map year to end-of-decade label: 1991-2000 -> 2000."""
    y = pd.to_numeric(series, errors="coerce")
    return ((y - 1) // 10 + 1) * 10


# %%
# Update any path below if your local folder names changed.
LED_PATH = '/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 1 - Exposure analysis/AGU Submission/AGU Earth Future - Initial Submission Package/data_release/US_final_population_building_inventory_pointData_with_socioeconomic-vMar12-2026.gpkg'
NHGIS_40_PATH = '/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics/Data/NHGIS/nhgis0004_COUNTY_csv/nhgis0004_ds78_1940_county.csv'
NHGIS_TS_PATH = '/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics/Data/NHGIS/nhgis0004_COUNTY_csv/nhgis0004_ts_nominal_county.csv'
SHAPE_40_PATH = '/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics/Data/NHGIS/nhgis0005_COUNTY_shape/nhgis0005_shapefile_tl2000_us_county_1940.zip'
SHAPE_2020_PATH = '/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics/Data/NHGIS/nhgis0005_COUNTY_shape/nhgis0005_shapefile_tl2020_us_county_2020.zip'

led_inventory = gpd.read_file(LED_PATH)
shape_40 = gpd.read_file(SHAPE_40_PATH)
shape_data = gpd.read_file(SHAPE_2020_PATH)
nhgis_40 = pd.read_csv(NHGIS_40_PATH, encoding="iso-8859-1")
nhgis_data = pd.read_csv(NHGIS_TS_PATH, encoding="iso-8859-1")

if "NHGISCODE" in nhgis_data.columns and "GISJOIN" not in nhgis_data.columns:
    nhgis_data = nhgis_data.rename(columns={"NHGISCODE": "GISJOIN"})
if "GISJOIN" not in nhgis_data.columns:
    raise ValueError("nhgis_data must contain GISJOIN or NHGISCODE")

if "NHGISCODE" in nhgis_40.columns and "GISJOIN" not in nhgis_40.columns:
    nhgis_40 = nhgis_40.rename(columns={"NHGISCODE": "GISJOIN"})
if "GISJOIN" not in nhgis_40.columns:
    raise ValueError("nhgis_40 must contain GISJOIN or NHGISCODE")

nhgis_data["GISJOIN"] = normalize_gisjoin(nhgis_data["GISJOIN"])
nhgis_40["GISJOIN"] = normalize_gisjoin(nhgis_40["GISJOIN"])
shape_40["GISJOIN"] = normalize_gisjoin(shape_40["GISJOIN"])
shape_data["GISJOIN"] = normalize_gisjoin(shape_data["GISJOIN"])

print(f"LED rows: {len(led_inventory):,}")
print(f"shape_40 rows: {len(shape_40):,}")
print(f"shape_data rows: {len(shape_data):,}")
print(f"nhgis_40 rows: {len(nhgis_40):,}")
print(f"nhgis_data rows: {len(nhgis_data):,}")


# %%
# Spatial joins with explicit spatial indexes (faster).
if "med_yr_blt" not in led_inventory.columns:
    raise ValueError("led_inventory is missing med_yr_blt")

if led_inventory.crs is None:
    raise ValueError("led_inventory has no CRS; cannot run spatial joins safely")

target_crs = led_inventory.crs
if shape_40.crs != target_crs:
    shape_40 = shape_40.to_crs(target_crs)
if shape_data.crs != target_crs:
    shape_data = shape_data.to_crs(target_crs)

shape_40_join = shape_40[["GISJOIN", "geometry"]].rename(columns={"GISJOIN": "GISJOIN_40"})
shape_data_join = shape_data[["GISJOIN", "geometry"]].rename(columns={"GISJOIN": "GISJOIN_2020"})

# Build spatial indexes before join.
_ = shape_40_join.sindex
_ = shape_data_join.sindex

led_joined = gpd.sjoin(
    led_inventory,
    shape_40_join,
    how="inner",
    predicate="intersects",
)
led_joined = led_joined.drop(columns=["index_right"], errors="ignore")

led_joined = gpd.sjoin(
    led_joined,
    shape_data_join,
    how="inner",
    predicate="intersects",
)
led_joined = led_joined.drop(columns=["index_right"], errors="ignore")

# 1) Use med_yr_blt to create decade (1991-2000 -> 2000).
led_joined["decade"] = decade_from_med_yr_blt(led_joined["med_yr_blt"])
led_joined = led_joined.dropna(subset=["decade", "GISJOIN_40", "GISJOIN_2020"]).copy()
led_joined["decade"] = led_joined["decade"].astype(int)
led_joined["GISJOIN_40"] = normalize_gisjoin(led_joined["GISJOIN_40"])
led_joined["GISJOIN_2020"] = normalize_gisjoin(led_joined["GISJOIN_2020"])

# 2) Aggregate led_count by GISJOIN and decade from earliest LED decade.
earliest_led_decade = int(led_joined["decade"].min())

led_counts_40_long = (
    led_joined[led_joined["decade"] >= earliest_led_decade]
    .groupby(["GISJOIN_40", "decade"], as_index=False)
    .size()
    .rename(columns={"size": "led_count"})
    .sort_values(["GISJOIN_40", "decade"])
)

led_counts_2020_long = (
    led_joined[led_joined["decade"] >= earliest_led_decade]
    .groupby(["GISJOIN_2020", "decade"], as_index=False)
    .size()
    .rename(columns={"size": "led_count"})
    .sort_values(["GISJOIN_2020", "decade"])
)

print(f"Rows after joins: {len(led_joined):,}")
print(f"Earliest LED decade: {earliest_led_decade}")


# %%
# 3) Merge cumulative count by GISJOIN_40 up to decade 1940 into nhgis_40.
led_40_long = (
    led_counts_40_long.copy()
)
led_40_long["led_cum_count"] = led_40_long.groupby("GISJOIN_40")["led_count"].cumsum()

led_cum_1940 = (
    led_40_long[led_40_long["decade"] <= 1940]
    .groupby("GISJOIN_40", as_index=False)["led_cum_count"]
    .max()
    .rename(columns={"GISJOIN_40": "GISJOIN", "led_cum_count": "led_cum_count_1940"})
)

nhgis_40["GISJOIN"] = normalize_gisjoin(nhgis_40["GISJOIN"])
led_cum_1940["GISJOIN"] = normalize_gisjoin(led_cum_1940["GISJOIN"])

# Keep reruns idempotent: remove previous merged output column before merging again.
if "led_cum_count_1940" in nhgis_40.columns:
    nhgis_40 = nhgis_40.drop(columns=["led_cum_count_1940"], errors="ignore")

nhgis_40 = nhgis_40.merge(led_cum_1940, on="GISJOIN", how="left")
nhgis_40["led_cum_count_1940"] = nhgis_40["led_cum_count_1940"].fillna(0).astype(int)


# %%
# 4) Merge cumulative count by GISJOIN_2020 for 1970-2020 into nhgis_data.
target_decades = [1970, 1980, 1990, 2000, 2010, 2020]

# Build cumulative columns across the full decade span first, then select reporting decades.
# This ensures led_cum_1970 includes all earlier decades (for example 1940+1950+1960+1970).
min_decade_for_cum = int((led_counts_2020_long["decade"].min() // 10) * 10)
cum_decades = list(range(min_decade_for_cum, max(target_decades) + 10, 10))

led_2020_counts_wide_all = (
    led_counts_2020_long[led_counts_2020_long["decade"] <= max(target_decades)]
    .pivot(index="GISJOIN_2020", columns="decade", values="led_count")
    .reindex(columns=cum_decades)
    .fillna(0)
)

led_2020_cum_wide_all = led_2020_counts_wide_all.cumsum(axis=1)
led_2020_wide = led_2020_cum_wide_all.reindex(columns=target_decades, fill_value=0)
led_2020_wide.columns = [f"led_cum_{int(c)}" for c in target_decades]
led_2020_wide = led_2020_wide.reset_index().rename(columns={"GISJOIN_2020": "GISJOIN"})

nhgis_data["GISJOIN"] = normalize_gisjoin(nhgis_data["GISJOIN"])
led_2020_wide["GISJOIN"] = normalize_gisjoin(led_2020_wide["GISJOIN"])

# Keep reruns idempotent: remove previous cumulative columns before merging again.
existing_led_cum_cols = [
    c for c in nhgis_data.columns
    if c.startswith("led_cum_") and c[8:].isdigit()
]
if existing_led_cum_cols:
    nhgis_data = nhgis_data.drop(columns=existing_led_cum_cols, errors="ignore")

nhgis_data = nhgis_data.merge(led_2020_wide, on="GISJOIN", how="left")

for d in target_decades:
    c = f"led_cum_{d}"
    if c in nhgis_data.columns:
        nhgis_data[c] = pd.to_numeric(nhgis_data[c], errors="coerce").fillna(0).astype(int)

print("Merged output tables ready")
print("nhgis_40 includes led_cum_count_1940")
print(f"nhgis_data includes {[f'led_cum_{d}' for d in target_decades]}")
print(f"nhgis_40 rows: {len(nhgis_40):,} | nhgis_data rows: {len(nhgis_data):,}")

# %%
# 5) Plot county-level decade differences and compute simple metrics.

decade_sources = {
    1940: (nhgis_40, "BXR001", "led_cum_count_1940"),
    1970: (nhgis_data, "A41AA1970", "led_cum_1970"),
    1980: (nhgis_data, "A41AA1980", "led_cum_1980"),
    1990: (nhgis_data, "A41AA1990", "led_cum_1990"),
    2000: (nhgis_data, "A41AA2000", "led_cum_2000"),
    2010: (nhgis_data, "A41AA2010", "led_cum_2010"),
    2020: (nhgis_data, "A41AA2020", "led_cum_2020"),
}

metrics_rows = []
decade_plot_data: dict[int, pd.DataFrame] = {}


def resolve_column(df: pd.DataFrame, base_col: str) -> str | None:
    """Resolve column names robustly across reruns (base, _y, _x)."""
    if base_col in df.columns:
        return base_col
    for suffix in ("_y", "_x"):
        cand = f"{base_col}{suffix}"
        if cand in df.columns:
            return cand
    return None

for decade, (src_df, nhgis_col, led_col) in decade_sources.items():
    nhgis_col_resolved = resolve_column(src_df, nhgis_col)
    led_col_resolved = resolve_column(src_df, led_col)
    if nhgis_col_resolved is None or led_col_resolved is None:
        print(f"Skipping {decade}: missing columns {nhgis_col} and/or {led_col}")
        continue

    plot_df = src_df[["GISJOIN", nhgis_col_resolved, led_col_resolved]].copy()
    plot_df[nhgis_col_resolved] = pd.to_numeric(plot_df[nhgis_col_resolved], errors="coerce")
    plot_df[led_col_resolved] = pd.to_numeric(plot_df[led_col_resolved], errors="coerce")
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[nhgis_col_resolved, led_col_resolved])

    if plot_df.empty:
        print(f"Skipping {decade}: no valid rows for comparison")
        continue

    plot_df = plot_df.rename(columns={nhgis_col_resolved: "nhgis_count", led_col_resolved: "led_count"})
    plot_df["error_diff"] = plot_df["led_count"] - plot_df["nhgis_count"]
    plot_df["abs_error"] = plot_df["error_diff"].abs()
    decade_plot_data[decade] = plot_df

    y_true = plot_df["nhgis_count"].to_numpy(dtype=float)
    y_pred = plot_df["led_count"].to_numpy(dtype=float)
    diff = plot_df["error_diff"].to_numpy(dtype=float)
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(plot_df) > 1 else np.nan
    rmse = float(np.sqrt(np.mean(diff**2)))

    metrics_rows.append(
        {
            "decade": decade,
            "n_counties": int(len(plot_df)),
            "mean_error_diff": float(np.mean(diff)),
            "corr": corr,
            "rmse": rmse,
        }
    )

metrics_by_decade = pd.DataFrame(metrics_rows).sort_values("decade").reset_index(drop=True)
print("County-level metrics by decade:")
print(metrics_by_decade)

decades_to_plot = [1940, 1970, 1980, 1990, 2000, 2010, 2020]
fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharex=False, sharey=False)
axes = axes.flatten()

for i, decade in enumerate(decades_to_plot):
    ax = axes[i]
    if decade not in decade_plot_data:
        ax.set_title(f"{decade}s (no valid data)")
        ax.grid(True, which="both", ls="--", linewidth=0.5)
        continue

    plot_df = decade_plot_data[decade]
    ax.scatter(plot_df["nhgis_count"], plot_df["error_diff"], alpha=0.5, s=12)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)

    m = metrics_by_decade.loc[metrics_by_decade["decade"] == decade].iloc[0]
    ax.set_title(
        f"{decade}s | n={int(m['n_counties'])} | corr={m['corr']:.2f} | rmse={m['rmse']:.1f}"
    )
    ax.set_xlabel("NHGIS built-up count")
    ax.set_ylabel("Difference (LED - NHGIS)")
    ax.grid(True, which="both", ls="--", linewidth=0.5)

for j in range(len(decades_to_plot), len(axes)):
    axes[j].axis("off")

fig.suptitle("County-level Difference by Decade (LED - NHGIS)", fontsize=14)
plt.tight_layout()
plt.show()

# %%
# 6) Plot cumullative time-series for the entire state of Washington
state_plot_data = []
for decade, (src_df, nhgis_col, led_col) in decade_sources.items():
    nhgis_col_resolved = resolve_column(src_df, nhgis_col)
    led_col_resolved = resolve_column(src_df, led_col)
    if nhgis_col_resolved is None or led_col_resolved is None:
        continue

    plot_df = src_df[["GISJOIN", nhgis_col_resolved, led_col_resolved]].copy()
    plot_df[nhgis_col_resolved] = pd.to_numeric(plot_df[nhgis_col_resolved], errors="coerce")
    plot_df[led_col_resolved] = pd.to_numeric(plot_df[led_col_resolved], errors="coerce")
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)

    # Keep county coverage consistent across decades for totals.
    # Missing values are treated as zero contribution to statewide sums.
    plot_df[nhgis_col_resolved] = plot_df[nhgis_col_resolved].fillna(0)
    plot_df[led_col_resolved] = plot_df[led_col_resolved].fillna(0)

    total_nhgis = plot_df[nhgis_col_resolved].sum()
    total_led = plot_df[led_col_resolved].sum()
    state_plot_data.append({"decade": decade, "total_nhgis": total_nhgis, "total_led": total_led})
state_plot_df = pd.DataFrame(state_plot_data).sort_values("decade").reset_index(drop=True)

if not state_plot_df.empty:
    led_non_decreasing = (state_plot_df["total_led"].diff().fillna(0) >= 0).all()
    print(f"LED statewide cumulative non-decreasing across plotted decades: {led_non_decreasing}")

plt.figure(figsize=(10, 6))
plt.plot(state_plot_df["decade"], state_plot_df["total_nhgis"], marker="o", label="NHGIS Total")
plt.plot(state_plot_df["decade"], state_plot_df["total_led"], marker="o", label="LED Total")
plt.title("National Total Built-up Area Count in the United States by Decade")
plt.xlabel("Decade")
plt.ylabel("Total Count")
plt.xticks(state_plot_df["decade"])
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# %%
# 7) Single combined 10x5 grid plot by state NAME (exclude Puerto Rico).
us_50_states = {
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
    "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
    "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi",
    "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico",
    "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
    "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
    "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"
}

state_long_rows = []
for decade, (src_df, nhgis_col, led_col) in decade_sources.items():
    nhgis_col_resolved = resolve_column(src_df, nhgis_col)
    led_col_resolved = resolve_column(src_df, led_col)
    if nhgis_col_resolved is None or led_col_resolved is None:
        continue

    state_col = "NAME" if "NAME" in src_df.columns else ("STATE" if "STATE" in src_df.columns else None)
    if state_col is None:
        print(f"Skipping {decade}: missing NAME/STATE column")
        continue

    state_df = src_df[[state_col, nhgis_col_resolved, led_col_resolved]].copy()
    state_df[state_col] = state_df[state_col].astype(str).str.strip().str.strip('"')
    state_df = state_df[state_df[state_col] != "Puerto Rico"]
    state_df = state_df[state_df[state_col].isin(us_50_states)]

    state_df[nhgis_col_resolved] = pd.to_numeric(state_df[nhgis_col_resolved], errors="coerce").fillna(0)
    state_df[led_col_resolved] = pd.to_numeric(state_df[led_col_resolved], errors="coerce").fillna(0)

    grouped = (
        state_df
        .groupby(state_col, as_index=False)[[nhgis_col_resolved, led_col_resolved]]
        .sum()
        .rename(columns={
            state_col: "state_name",
            nhgis_col_resolved: "total_nhgis",
            led_col_resolved: "total_led",
        })
    )
    grouped["decade"] = decade
    state_long_rows.append(grouped)

if not state_long_rows:
    raise RuntimeError("No state-level rows were built for the 10x5 plot.")

state_long = pd.concat(state_long_rows, ignore_index=True)
states_to_plot = sorted(state_long["state_name"].dropna().unique().tolist())

if len(states_to_plot) != 50:
    print(f"Warning: expected 50 states, found {len(states_to_plot)} in data.")

plot_decades = [1940, 1970, 1980, 1990, 2000, 2010, 2020]

fig, axes = plt.subplots(10, 5, figsize=(24, 34), sharex=True, sharey=True)
axes = axes.flatten()

for i, state_name in enumerate(states_to_plot[:50]):
    ax = axes[i]
    sdf = state_long[state_long["state_name"] == state_name].copy()
    sdf = sdf.sort_values("decade")
    sdf = sdf[sdf["decade"].isin(plot_decades)]

    ax.plot(sdf["decade"], sdf["total_nhgis"], marker="o", linewidth=1.2, label="NHGIS")
    ax.plot(sdf["decade"], sdf["total_led"], marker="o", linewidth=1.2, label="LED")
    ax.set_title(state_name, fontsize=9)
    ax.grid(True, which="both", ls="--", linewidth=0.4)

for j in range(min(len(states_to_plot), 50), len(axes)):
    axes[j].axis("off")

for ax in axes:
    ax.set_xticks(plot_decades)

fig.suptitle("State-Level Cumulative Built-up Count by Decade (50 U.S. States)", fontsize=16)
fig.supxlabel("Decade")
fig.supylabel("Count")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.99))
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
# %%
