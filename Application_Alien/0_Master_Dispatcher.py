# Master Dispatcher Script for Landslide Exposure Dynamics Analysis
# This script orchestrates the execution of the entire pipeline for a list of assigned states.

import subprocess
import time
from pathlib import Path
import datetime
import sys

# --- 1. Node Configuration ---
# Split states here depending on which computer is running the script.
# Node 1 Example (20 states): ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD']
# Node 2 Example (20 states): ['MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC']
# Node 3 Example (11 states): ['SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC']

ASSIGNED_STATES = ['OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'] # States in the Alien

# --- 2. Paths to your Scripts ---
SCRIPT_DIR = Path(r"C:\Users\danie\OneDrive - UW\0 - DA General Exam\Paper 2 - Temporal Dynamics\Analysis\Landslide_Exposure_Dynamics\Application_Alien")
SCRIPT_1 = SCRIPT_DIR / "1_Prod_Time-Matrix_HISDAC_UPDATED_Apr27.py"
SCRIPT_2 = SCRIPT_DIR / "2_Prod_Hierarchical_Tiers_DetProbDist_Apr27.py"
SCRIPT_3 = SCRIPT_DIR / "3_Prod_Visualization_Time-Matrix_HISDAC_Apr27.py"
SCRIPT_4 = SCRIPT_DIR / "4_Prod_MacroValidation_NHGIS_Apr27.py"
PYTHON_EXECUTABLE = sys.executable

def run_script(script_path: Path, state: str):
    """Executes a python script with the state argument and checks for errors."""
    print(f"      -> Running {script_path.name}...")
    result = subprocess.run([PYTHON_EXECUTABLE, str(script_path), "--state", state], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"\n[ERROR] in {script_path.name} for {state}:")
        print(result.stderr)
        raise RuntimeError(f"Pipeline halted for {state} due to script error.")
    return result.stdout

# --- 3. Execution Loop ---
print(f"=== STARTING BATCH RUN FOR {len(ASSIGNED_STATES)} STATES ===")
print(f"Python interpreter: {PYTHON_EXECUTABLE}")
global_start_time = time.time()

for state in ASSIGNED_STATES:
    print(f"\n========================================")
    print(f"Processing State: {state}")
    print(f"========================================")
    state_start_time = time.time()
    
    try:
        # Sequentially run the pipeline
        run_script(SCRIPT_1, state)
        run_script(SCRIPT_2, state)
        run_script(SCRIPT_3, state)
        run_script(SCRIPT_4, state)

        state_end_time = time.time()
        elapsed_min = (state_end_time - state_start_time) / 60
        print(f"   [SUCCESS] {state} completed in {elapsed_min:.2f} minutes.")
        
    except Exception as e:
        print(f"   [FAILED] {state} skipped. Error: {e}")
        continue # Move to the next state even if one fails

global_end_time = time.time()
total_elapsed_hours = (global_end_time - global_start_time) / 3600
print(f"\n=== BATCH RUN COMPLETE ===")
print(f"Total processing time: {total_elapsed_hours:.2f} hours.")