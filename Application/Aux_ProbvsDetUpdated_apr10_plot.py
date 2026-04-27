# %% [markdown]
# # National Exposure Trajectories (Dual-Model Plotter)
# #### Daniel Acosta-Reyes
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# --- 1. Setup Paths & Configuration ---
# ==============================================================================
print("1) Initializing Dual-Model National Plotter...")

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
hazard_classes = ['high', 'moderate', 'low', 'none']
color_map = {'high': '#d62728', 'moderate': '#ff7f0e', 'low': '#f1c40f', 'none': '#7f8c8d'}

# National Accumulators
national_cum_P = {c: np.zeros(len(YEARS)) for c in hazard_classes}
national_cum_D = {c: np.zeros(len(YEARS)) for c in hazard_classes}
all_states_ts_data = []

# ==============================================================================
# --- 2. Master Loop: Calculate & Plot Every State ---
# ==============================================================================
for STATE_CODE in STATES_LOWER_48:
    PROD_PATH = DATA_PATH / "Production_set" / STATE_CODE
    ANALYTICS_PATH = ANALYSIS_PATH / "Production_analytics" / STATE_CODE
    
    engine_file = PROD_PATH / f"{STATE_CODE}_LED_Monte_Carlo_Engine.parquet"
    master_file = PROD_PATH / f"{STATE_CODE}_Master_Spatiotemporal_Matrix.parquet"
    
    if not engine_file.exists() or not master_file.exists():
        continue
        
    print(f"\nProcessing {STATE_CODE}...")
    ANALYTICS_PATH.mkdir(parents=True, exist_ok=True)
    
    # --- A. Load Probabilistic Engine (Fast Array Extraction) ---
    led_df = pd.read_parquet(engine_file)
    led_df['susc_class'] = led_df['susc_class'].astype(str).str.lower().str.strip()
    
    prob_matrix = np.stack(led_df['prob_distribution'].values)
    cum_prob_matrix = np.cumsum(prob_matrix, axis=1)
    
    cum_P = {}
    for c in hazard_classes:
        mask = led_df['susc_class'] == c
        if mask.sum() > 0:
            cum_P[c] = np.sum(cum_prob_matrix[mask], axis=0)
        else:
            cum_P[c] = np.zeros(len(YEARS))
            
    # --- B. Calculate Updated Deterministic Math ---
    master_df = pd.read_parquet(master_file)
    led_exposed_counts = led_df.groupby(['HISDAC_id', 'susc_class']).size().unstack(fill_value=0)
    det_matrix = master_df.join(led_exposed_counts, on='HISDAC_id', how='left')

    for c in hazard_classes:
        if c not in det_matrix.columns: det_matrix[c] = 0
    det_matrix[hazard_classes] = det_matrix[hazard_classes].fillna(0)

    det_matrix['Total_LED_2020'] = det_matrix[hazard_classes].sum(axis=1)
    det_matrix['Total_HISDAC'] = det_matrix['C_BUPL2020'].fillna(0)
    det_matrix['Cumulative_LED_Fallback'] = 0

    cum_D = {c: np.zeros(len(YEARS)) for c in hazard_classes}
    
    for y_idx, y in enumerate(YEARS):
        cumulative_hisdac = det_matrix[f'C_BUPL{y}'].fillna(0).values
        if f'LED_{y}' in det_matrix.columns:
            det_matrix['Cumulative_LED_Fallback'] += det_matrix[f'LED_{y}'].fillna(0)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            r_hisdac = np.where(det_matrix['Total_HISDAC'] > 0, cumulative_hisdac / det_matrix['Total_HISDAC'], 0)
            r_led = np.where(det_matrix['Total_LED_2020'] > 0, det_matrix['Cumulative_LED_Fallback'] / det_matrix['Total_LED_2020'], 0)
            
        ratio = np.where(det_matrix['Total_HISDAC'] > 0, r_hisdac, r_led)
        if y == 2020: ratio = 1.0
            
        for c in hazard_classes:
            cum_D[c][y_idx] = (det_matrix[c].values * ratio).sum()

    # --- C. Calculate Rates & Add to National Aggregate ---
    tot_built_P = np.zeros(len(YEARS))
    tot_built_D = np.zeros(len(YEARS))
    
    for c in hazard_classes:
        tot_built_P += cum_P[c]
        tot_built_D += cum_D[c]
        national_cum_P[c] += cum_P[c]
        national_cum_D[c] += cum_D[c]

    rate_P = {}
    rate_D = {}
    for c in hazard_classes:
        with np.errstate(divide='ignore', invalid='ignore'):
            rate_P[c] = np.where(tot_built_P > 0, (cum_P[c] / tot_built_P) * 100, 0)
            rate_D[c] = np.where(tot_built_D > 0, (cum_D[c] / tot_built_D) * 100, 0)
            
        # Store Data for the CSV Output
        for y_idx, y in enumerate(YEARS):
            all_states_ts_data.append({
                'State': STATE_CODE, 'Year': y, 'Hazard': c,
                'Prob_Cum': cum_P[c][y_idx], 'Det_Cum': cum_D[c][y_idx],
                'Prob_Rate': rate_P[c][y_idx], 'Det_Rate': rate_D[c][y_idx]
            })

    # --- D. Plot the State Comparative Exposure Rate ---
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(f"Landslide Exposure Rate E(t): Probabilistic vs Deterministic ({STATE_CODE})", fontsize=16, fontweight='bold')

    for c in ['high', 'moderate', 'low']:
        ax.plot(YEARS, rate_P[c], color=color_map[c], linestyle='-', linewidth=3, marker='o', label=f"{c.capitalize()} Rate (Probabilistic)")
        ax.plot(YEARS, rate_D[c], color=color_map[c], linestyle='--', linewidth=2, alpha=0.7, label=f"{c.capitalize()} Rate (Deterministic)")

    ax.set_title("E(t) = Exposed Buildings / Total Buildings (With Smoothed 2020 Deterministic Fallback)", fontsize=12)
    ax.set_ylabel("Exposure Rate (%)")
    ax.set_xlabel("Year")
    ax.set_xticks(YEARS)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left', ncol=2)

    fig.tight_layout()
    fig.savefig(ANALYTICS_PATH / f"{STATE_CODE}_Exposure_Rate_Dual_Comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

# ==============================================================================
# --- 3. Generate National Aggregate Plot & Export ---
# ==============================================================================
print("\n3) Generating National Aggregate Analytics...")

nat_tot_built_P = np.zeros(len(YEARS))
nat_tot_built_D = np.zeros(len(YEARS))

for c in hazard_classes:
    nat_tot_built_P += national_cum_P[c]
    nat_tot_built_D += national_cum_D[c]

nat_rate_P = {}
nat_rate_D = {}

for c in hazard_classes:
    with np.errstate(divide='ignore', invalid='ignore'):
        nat_rate_P[c] = np.where(nat_tot_built_P > 0, (national_cum_P[c] / nat_tot_built_P) * 100, 0)
        nat_rate_D[c] = np.where(nat_tot_built_D > 0, (national_cum_D[c] / nat_tot_built_D) * 100, 0)

# National Exposure Rate Plot
fig_nat, ax_nat = plt.subplots(figsize=(14, 8))
fig_nat.suptitle("NATIONAL Landslide Exposure Rate E(t): Probabilistic vs Deterministic (Lower 48)", fontsize=16, fontweight='bold')

for c in ['high', 'moderate', 'low']:
    ax_nat.plot(YEARS, nat_rate_P[c], color=color_map[c], linestyle='-', linewidth=3, marker='o', label=f"{c.capitalize()} Rate (Probabilistic)")
    ax_nat.plot(YEARS, nat_rate_D[c], color=color_map[c], linestyle='--', linewidth=2, alpha=0.7, label=f"{c.capitalize()} Rate (Deterministic)")

ax_nat.set_title("Aggregated Exposure Across the Continental United States (1920-2020)", fontsize=12)
ax_nat.set_ylabel("National Exposure Rate (%)")
ax_nat.set_xlabel("Year")
ax_nat.set_xticks(YEARS)
ax_nat.tick_params(axis='x', rotation=45)
ax_nat.grid(True, linestyle='--', alpha=0.6)
ax_nat.legend(loc='upper left', ncol=2)

fig_nat.tight_layout()
fig_nat.savefig(NATIONAL_OUT / "US_National_Exposure_Rate_Comparison.png", dpi=300, bbox_inches='tight')
plt.close(fig_nat)

# Save the time-series CSV for all states
ts_df = pd.DataFrame(all_states_ts_data)
ts_df.to_csv(NATIONAL_OUT / "All_States_TimeSeries_Exposure.csv", index=False)

print(f"\n=== SCRIPT COMPLETE: National Plot and Time-Series CSV saved to {NATIONAL_OUT} ===")

# %% [markdown]
# # All-States Spaghetti Plotter (National Variance)
# #### Daniel Acosta-Reyes
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# --- 1. Setup Paths ---
# ==============================================================================
print("1) Loading National Master Dataset...")

ANALYSIS_PATH = Path("/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics/Analysis")
NATIONAL_OUT = ANALYSIS_PATH / "National_Summary"

# Load the CSV generated by your master loop
csv_path = NATIONAL_OUT / "All_States_TimeSeries_Exposure.csv"

if not csv_path.exists():
    print(f"[ERROR] Could not find {csv_path}. Run the Master Plotter first.")
    exit()

df = pd.read_csv(csv_path)

YEARS = sorted(df['Year'].unique())
STATES = df['State'].unique()
hazard_classes = ['high', 'moderate', 'low']
color_map = {'high': '#d62728', 'moderate': '#ff7f0e', 'low': '#f1c40f'}

# ==============================================================================
# --- 2. Calculate True National Averages ---
# ==============================================================================
# To get the true national rate, we must sum the cumulative volumes first, 
# then calculate the rate. We cannot simply average the state percentages!
print("2) Calculating True National Averages...")

nat_agg = df.groupby(['Year', 'Hazard'])[['Prob_Cum', 'Det_Cum']].sum().reset_index()

# We need the total built environment per year to calculate the rate
# Total built = sum of high + moderate + low + none
yearly_totals_prob = df.groupby(['Year', 'State'])['Prob_Cum'].sum().groupby('Year').sum()
yearly_totals_det = df.groupby(['Year', 'State'])['Det_Cum'].sum().groupby('Year').sum()

nat_agg['Prob_Rate'] = nat_agg.apply(lambda row: (row['Prob_Cum'] / yearly_totals_prob[row['Year']] * 100) if yearly_totals_prob[row['Year']] > 0 else 0, axis=1)
nat_agg['Det_Rate'] = nat_agg.apply(lambda row: (row['Det_Cum'] / yearly_totals_det[row['Year']] * 100) if yearly_totals_det[row['Year']] > 0 else 0, axis=1)

# ==============================================================================
# --- 3. Plotting Function (Spaghetti Plot) ---
# ==============================================================================
def plot_all_states_spaghetti(model_type='Probabilistic'):
    print(f"3) Generating {model_type} All-States Plot...")
    
    rate_col = 'Prob_Rate' if model_type == 'Probabilistic' else 'Det_Rate'
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    fig.suptitle(f"National Variance in Landslide Exposure Rates: All States ({model_type} Model)", fontsize=18, fontweight='bold', y=1.05)
    
    for i, hazard in enumerate(hazard_classes):
        ax = axes[i]
        color = color_map[hazard]
        
        # Plot every individual state as a thin, semi-transparent line
        for state in STATES:
            state_data = df[(df['State'] == state) & (df['Hazard'] == hazard)].sort_values('Year')
            ax.plot(state_data['Year'], state_data[rate_col], color='gray', alpha=0.15, linewidth=1)
            
            # Optional: Highlight specific states of interest (e.g., WA, CA, WV)
            if state in ['WA', 'CA', 'WV']:
                ax.plot(state_data['Year'], state_data[rate_col], label=f"{state} Trend", linewidth=1.5, alpha=0.8, linestyle=':')

        # Plot the True National Average as a thick bold line
        nat_data = nat_agg[nat_agg['Hazard'] == hazard].sort_values('Year')
        ax.plot(nat_data['Year'], nat_data[rate_col], color=color, linewidth=4, label='National Average')
        
        ax.set_title(f"{hazard.capitalize()} Hazard Zone", fontsize=14, fontweight='bold')
        ax.set_xlabel("Year", fontsize=12)
        if i == 0:
            ax.set_ylabel("Exposure Rate (%)", fontsize=12)
        ax.set_xticks(YEARS[::2]) # Show every other year to avoid crowding
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle='--', alpha=0.4)
        
        # Only show legend for the last subplot to save space
        if i == 2:
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    plt.tight_layout()
    output_filename = f"All_States_Spaghetti_{model_type}.png"
    fig.savefig(NATIONAL_OUT / output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   -> Saved: {output_filename}")

# ==============================================================================
# --- 4. Execute Plots ---
# ==============================================================================
plot_all_states_spaghetti(model_type='Probabilistic')
plot_all_states_spaghetti(model_type='Deterministic')

print(f"\n=== ALL-STATES PLOTS COMPLETE. Saved to {NATIONAL_OUT} ===")