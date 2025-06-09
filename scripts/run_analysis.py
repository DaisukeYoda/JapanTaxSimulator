import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Attempt to import necessary modules from the project
try:
    from src.dsge_model import DSGEModel, ModelParameters
    from src.tax_simulator import EnhancedTaxSimulator, TaxReform # Use EnhancedTaxSimulator directly
    # from src.plot_utils import plot_impulse_responses # This might need to be adapted or implemented
    # from src.utils import ensure_dir_exists # This needs to be implemented, currently in script
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure that the src directory is in PYTHONPATH or the script is run from the project root.")
    # Try adding project root to PYTHONPATH if running from scripts/
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    try:
        from src.dsge_model import DSGEModel, ModelParameters
        from src.tax_simulator import EnhancedTaxSimulator, TaxReform # Use EnhancedTaxSimulator directly
        # from src.plot_utils import plot_impulse_responses # This might need to be adapted or implemented
        # from src.utils import ensure_dir_exists # This needs to be implemented
    except ImportError:
        print("Failed to import after attempting to modify PYTHONPATH.")
        raise

# --- Configuration ---
OUTPUT_SUBDIR = "results/current_analysis_report_data"
# ensure_dir_exists(OUTPUT_SUBDIR) # Will implement this later

SIMULATION_PERIODS = 40 # Standard duration for IRFs

# --- Helper Functions ---
# Implementation for ensure_dir_exists
def ensure_dir_exists(directory_path: str):
    """Ensures that a directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    print(f"Directory {directory_path} ensured.")

ensure_dir_exists(OUTPUT_SUBDIR)


def save_simulation_results(reform_name: str, results, steady_state_comparison: pd.DataFrame): # results type updated
    """Saves simulation results (IRFs and steady state) to CSV files."""
    # Assuming results is now a SimulationResults object
    # IRFs are in reform_path (or potentially baseline_path vs reform_path)
    irf_df = results.reform_path # Corrected: Access reform_path from SimulationResults
    irf_df.to_csv(os.path.join(OUTPUT_SUBDIR, f"irf_{reform_name}.csv"))

    steady_state_comparison.to_csv(os.path.join(OUTPUT_SUBDIR, f"ss_comp_{reform_name}.csv"))
    print(f"Results saved for {reform_name}")

def format_steady_state_comparison(baseline_ss, reform_ss, variables_to_compare):
    """Formats steady state comparison into a DataFrame."""
    data = []
    for var in variables_to_compare:
        val_baseline = getattr(baseline_ss, var, np.nan)
        val_reform = getattr(reform_ss, var, np.nan)
        change = (val_reform - val_baseline)
        pct_change = ((val_reform / val_baseline) - 1) * 100 if val_baseline != 0 and not (np.isnan(val_baseline) or np.isnan(val_reform)) else np.nan
        data.append({
            "Variable": var,
            "Baseline": val_baseline,
            "Reform": val_reform,
            "Absolute Change": change,
            "Percentage Change": pct_change
        })
    return pd.DataFrame(data)

# --- Main Simulation Logic ---
def run_simulations():
    """Runs a predefined set of tax policy simulations."""
    print("Starting tax policy simulations...")

    # Load model parameters from JSON
    try:
        # Assuming script is run from project root, or scripts/
        param_file = "config/parameters.json"
        if not os.path.exists(param_file) and os.path.exists(os.path.join("..", param_file)):
            param_file = os.path.join("..", param_file)
        params = ModelParameters.from_json(param_file)
    except FileNotFoundError:
        print(f"Error: {param_file} not found. Ensure you are in the project root or scripts/ directory.")
        return
    except Exception as e:
        print(f"Error loading parameters: {e}")
        return

    # Initialize model and simulator
    try:
        baseline_model = DSGEModel(params)

        # Calculate baseline steady state FIRST
        print("Calculating baseline steady state...")
        baseline_ss = baseline_model.compute_steady_state() # This now sets baseline_model.steady_state
        print("Baseline steady state calculated successfully.")

        # Then initialize simulator
        # Using EnhancedTaxSimulator explicitly as TaxPolicySimulator might be an old alias
        simulator = EnhancedTaxSimulator(baseline_model) # Default research_mode=False, use_simple_model=False

    except Exception as e:
        # This will catch errors from both DSGEModel init, compute_steady_state, and EnhancedTaxSimulator init
        print(f"Error during model or simulator initialization or baseline steady state computation: {e}")
        return

    # Define variables for steady state comparison
    ss_vars_to_compare = ['Y', 'C', 'I', 'L', 'G', 'T_total_revenue', 'B', 'w', 'Rk_gross', 'pi_gross'] # Corrected: W to w, R_k to Rk_gross, Pi to pi_gross


    # --- Define Tax Reform Scenarios ---
    reforms_to_simulate = []

    # 1. Consumption Tax Increase (Permanent)
    reforms_to_simulate.append(TaxReform(
        name="consumption_tax_increase_5pp_permanent",
        # description="Consumption tax +5pp (10% to 15%), permanent", # Description not a field in TaxReform
        tau_c=0.15, # Corrected: tau_c_new to tau_c
        implementation='permanent' # Corrected: implementation_type to implementation
    ))

    # 2. Income Tax Reduction (Permanent)
    reforms_to_simulate.append(TaxReform(
        name="income_tax_reduction_5pp_permanent",
        # description="Labor income tax -5pp (20% to 15%), permanent",
        tau_l=0.15, # Corrected: tau_l_new to tau_l
        implementation='permanent'
    ))

    # 3. Consumption Tax Increase (Phased)
    reforms_to_simulate.append(TaxReform(
        name="consumption_tax_increase_5pp_phased",
        # description="Consumption tax +5pp (10% to 15%), phased over 2 years (8 quarters)",
        tau_c=0.15,
        implementation='phased',
        phase_in_periods=8
    ))

    # 4. Revenue-Neutral: Consumption Tax Up, Income Tax Down
    reforms_to_simulate.append(TaxReform(
        name="revenue_neutral_c_up_l_down",
        # description="Revenue Neutral: Cons. tax +2pp, Income tax -2pp (approx.)",
        tau_c=0.12,
        tau_l=0.18,
        implementation='permanent'
    ))

    # --- Run Simulations for Each Reform ---
    for reform in reforms_to_simulate:
        print(f"--- Simulating: {reform.name} ---") # Removed reform.description as it's not a field
        try:
            # Simulate reform
            # TaxPolicySimulator.simulate_reform now returns a SimulationResults object
            simulation_output = simulator.simulate_reform(reform, periods=SIMULATION_PERIODS)

            # Get reform_ss from the SimulationResults object
            reform_ss = simulation_output.steady_state_reform

            # Prepare steady state comparison
            ss_comparison_df = format_steady_state_comparison(baseline_ss, reform_ss, ss_vars_to_compare)

            # Save results
            save_simulation_results(reform.name, simulation_output, ss_comparison_df)

            print(f"Simulation successful for {reform.name}.")

            # Optional: Plot and save IRFs as images
            # Ensure plot_utils.plot_impulse_responses is compatible
            # The dummy plot_impulse_responses will be used for now.
            # simulation_output.reform_path contains the IRFs (as a DataFrame)
            if 'plot_impulse_responses' in globals() and callable(plot_impulse_responses):
                try:
                    # Select some key variables to plot, ensure they exist in the IRF data
                    irf_df_plot = simulation_output.reform_path
                    plot_vars = [v for v in ['Y', 'C', 'I', 'L'] if v in irf_df_plot.columns]
                    if plot_vars:
                         fig = plot_impulse_responses(irf_df_plot, reform.name, plot_vars)
                         if fig: # If plot_impulse_responses returns a figure
                            plt.savefig(os.path.join(OUTPUT_SUBDIR, f"irf_plot_{reform.name}.png"))
                            plt.close(fig)
                    else:
                        print(f"Skipping plot for {reform.name}, no key variables in IRF data.")
                except Exception as plot_e:
                    print(f"Plotting failed for {reform.name}: {plot_e}")


        except NotImplementedError as nie:
            print(f"NotImplementedError during simulation for {reform.name}: {nie}")
        except Exception as e:
            print(f"Error during simulation for {reform.name}: {e}")

    print("All simulations attempted.")

# --- Global plot_utils for script execution ---
# Create a dummy plot_utils if not available, for basic script execution
# This will be replaced by actual import if available
_plot_impulse_responses_defined = 'plot_impulse_responses' in globals() and callable(plot_impulse_responses)

if not _plot_impulse_responses_defined:
    def plot_impulse_responses(irf_df, title, variables, shock_name=None):
        print(f"Dummy plot_impulse_responses called for {title} with variables {variables}. No actual plotting.")
        # Create a dummy figure to allow saving
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Plot for {title}\n(Dummy)", ha='center', va='center')
        return fig # Return a figure object so savefig can be called

if __name__ == "__main__":
    run_simulations()
    print(f"Simulation script finished. Check {OUTPUT_SUBDIR} for results.")
