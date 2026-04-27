# %%
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# --- 1. Flexible State Selection ---
state_code = input("Enter the 2-letter state code (e.g., WV, WA, AL): ").strip().upper()

# --- 2. Setup Paths ---
ANALYSIS_PATH = Path("/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics/Analysis")
CSV_PATH = ANALYSIS_PATH / "Production_analytics" / state_code / f"{state_code}_State_Exposure_Story.csv"

if not CSV_PATH.exists():
    print(f"\n[ERROR] Could not find the CSV file at:\n{CSV_PATH}")
    print("Please ensure the pipeline has finished running for this state.")
    sys.exit(1)

print(f"\nLoading data for {state_code}...")
df = pd.read_csv(CSV_PATH)

# --- 3. Data Aggregation (Combining High & Moderate) ---
# Filter to only keep the two hazard classes we care about
df_hm = df[df['Hazard'].isin(['high', 'moderate'])]

# Group by Year and Method, and sum the rates. 
# (Since High and Moderate share the exact same denominator per year/method, summing the % is mathematically perfectly accurate)
combined_df = df_hm.groupby(['Year', 'Method'])[['Cum_Rate_%', 'Marginal_Rate_%']].sum().reset_index()

# Split into two dataframes for clean plotting
prob_df = combined_df[combined_df['Method'] == 'Probabilistic'].sort_values('Year')
det_df = combined_df[combined_df['Method'] == 'Deterministic'].sort_values('Year')

# --- 4. Visualization ---
print("Generating Combined Exposure Plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f"Combined Landslide Exposure Rates (High + Moderate): {state_code}", fontsize=18, fontweight='bold')

# Colors matching our established palette (using a deep red/orange for the combined hazard)
color_p = '#d62728' # Solid Red for Probabilistic
color_d = '#ff7f0e' # Orange Dashed for Deterministic

# --- Subplot 1: Cumulative Rate ---
ax1.plot(prob_df['Year'], prob_df['Cum_Rate_%'], color=color_p, linestyle='-', linewidth=3, marker='o', label='Probabilistic (Building-Level)')
ax1.plot(det_df['Year'], det_df['Cum_Rate_%'], color=color_d, linestyle='--', linewidth=2.5, marker='^', alpha=0.8, label='Deterministic (Area-Level Baseline)')

ax1.set_title("Cumulative Exposure Rate (% of Total Standing Buildings)", fontsize=14)
ax1.set_ylabel("Exposure Rate (%)", fontsize=12)
ax1.set_xlabel("Year", fontsize=12)
ax1.set_xticks(prob_df['Year'])
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='best')

# --- Subplot 2: Marginal Rate ---
ax2.plot(prob_df['Year'], prob_df['Marginal_Rate_%'], color=color_p, linestyle='-', linewidth=3, marker='o', label='Probabilistic (Building-Level)')
ax2.plot(det_df['Year'], det_df['Marginal_Rate_%'], color=color_d, linestyle='--', linewidth=2.5, marker='^', alpha=0.8, label='Deterministic (Area-Level Baseline)')

ax2.set_title("Marginal Exposure Rate (% of NEW Construction per Interval)", fontsize=14)
ax2.set_ylabel("Marginal Rate (%)", fontsize=12)
ax2.set_xlabel("Year", fontsize=12)
ax2.set_xticks(prob_df['Year'])
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='best')

plt.tight_layout()
plt.show()

# Optional: Save the plot to the analytics folder
save_path = ANALYSIS_PATH / "Production_analytics" / state_code / f"{state_code}_Combined_HighMod_Rates.png"
fig.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {save_path}")