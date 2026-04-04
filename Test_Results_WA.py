# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Setup Paths ---
DATA_PATH = Path("/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics/Data")
TESTING_SET_PATH = DATA_PATH / "Testing_Set"
state_code = "WA"

print("1) Loading Monte Carlo Results...")
results_file = TESTING_SET_PATH / f"{state_code}_Monte_Carlo_Exposure_Results.csv"
df = pd.read_csv(results_file)

# --- 2) Data Transformation ---
# Pivot the dataframe so Years are rows and Hazard Classes are columns (using the Mean Expected Buildings)
cumulative_df = df.pivot(index='Year', columns='Hazard_Class', values='Mean_Expected_Buildings')

# Ensure column order makes sense for plotting
hazard_order = ['none', 'low', 'moderate', 'high']
cumulative_df = cumulative_df[hazard_order]

# Calculate Total Cumulative Buildings for each year
cumulative_df['Total_Built'] = cumulative_df.sum(axis=1)

# Calculate Deltas (New buildings added in each 5-year interval)
# diff() calculates current row minus previous row. Fill the 1920 NaN with the 1920 baseline.
marginal_df = cumulative_df[hazard_order].diff().fillna(cumulative_df[hazard_order].iloc[0])
marginal_df['Total_New_Built'] = marginal_df.sum(axis=1)

# --- 3) Calculate Proportions (%) ---
# 1. Cumulative Proportion (% of all standing buildings)
cum_prop_df = cumulative_df[hazard_order].div(cumulative_df['Total_Built'], axis=0) * 100

# 2. Marginal Proportion (% of new construction in that specific interval)
# Avoid division by zero in case a year had absolutely 0 new buildings (rare, but mathematically safe)
marg_prop_df = marginal_df[hazard_order].div(marginal_df['Total_New_Built'].replace(0, np.nan), axis=0) * 100

# --- 4) Visualizations ---
print("2) Generating Proportional Analytics...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True, dpi=300)
fig.suptitle(f"Proportional Landslide Exposure Dynamics: {state_code}", fontsize=16, fontweight='bold')

color_map = {'high': '#d62728', 'moderate': '#ff7f0e', 'low': '#f1c40f', 'none': '#bdc3c7'}
colors = [color_map[h] for h in hazard_order]
labels = [h.capitalize() for h in hazard_order]
years = cumulative_df.index

# Plot 1: 100% Stacked Area Chart (Cumulative Legacy)
ax1.stackplot(years, cum_prop_df.T, labels=labels, colors=colors, alpha=0.85)
ax1.set_title('Cumulative Exposure Share (% of Total Built Environment)', fontsize=14)
ax1.set_ylabel('Percentage (%)')
ax1.set_ylim(0, 100)
# Reverse legend order so 'High' is at the top of the legend, matching the visual stack
handles, leg_labels = ax1.get_legend_handles_labels()
ax1.legend(handles[::-1], leg_labels[::-1], loc='center left', bbox_to_anchor=(1, 0.5))
ax1.grid(axis='x', linestyle='--', alpha=0.6)

# Plot 2: Marginal Risk Lines (Active Decisions)
# We plot lines instead of a stacked chart here to easily see peaks and valleys
for hazard in ['high', 'moderate', 'low']: # Skipping 'none' to zoom in on the risk
    ax2.plot(years, marg_prop_df[hazard], marker='o', linewidth=2.5, 
             color=color_map[hazard], label=f"{hazard.capitalize()} Landslide Susceptibility")

ax2.set_title('Marginal Exposure Rate (% of NEW Construction per Interval)', fontsize=14)
ax2.set_xlabel('Year')
ax2.set_ylabel('% of New Buildings Built in Landslide-Susceptible Zone')
ax2.legend(loc='upper left')
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0, 0.95, 0.96]) # Adjust rect to fit the external legend
plt.show()

# --- 5) Quick Console Insights ---
print("\n--- Key Statistical Insights ---")
latest_year = years[-1]
print(f"In {latest_year}, High Landslide Susceptibility represented {cum_prop_df.loc[latest_year, 'high']:.2f}% of the TOTAL state inventory.")

peak_marginal_year = marg_prop_df['high'].idxmax()
peak_marginal_val = marg_prop_df['high'].max()
print(f"The most dangerous era for development was {peak_marginal_year}, where {peak_marginal_val:.2f}% of all NEW construction occurred in High Landslide Susceptible zones.")

# %%
