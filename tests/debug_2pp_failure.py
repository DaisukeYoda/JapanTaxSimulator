"""
Detailed analysis of why 2pp consumption tax fails while 1pp and 5pp succeed
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dsge_model import DSGEModel, ModelParameters

def analyze_2pp_failure():
    """Analyze why specifically 2pp consumption tax fails"""
    print("=" * 70)
    print("2PP CONSUMPTION TAX FAILURE ANALYSIS")
    print("=" * 70)
    
    # Get baseline
    params = ModelParameters.from_json('config/parameters.json')
    baseline_model = DSGEModel(params)
    baseline_ss = baseline_model.compute_steady_state()
    
    print("Baseline steady state computed successfully")
    
    # Test different consumption tax levels around 2pp
    tax_levels = [0.105, 0.11, 0.115, 0.12, 0.125, 0.13]  # 0.5pp to 3pp
    
    print(f"\nTesting consumption tax levels around 2pp:")
    print(f"{'Tax Rate':<10} {'Change':<8} {'Strategy':<20} {'Result':<15} {'Max Residual':<15}")
    print("-" * 70)
    
    for tau_c in tax_levels:
        change_pp = (tau_c - 0.10) * 100
        
        # Create reform parameters
        reform_params = ModelParameters.from_json('config/parameters.json')
        reform_params.tau_c = tau_c
        reform_model = DSGEModel(reform_params)
        
        # Calculate strategy that will be used
        tax_change_magnitude = abs(tau_c - 0.10)
        if tax_change_magnitude < 0.03:
            strategy = "Baseline values"
        else:
            strategy = "Tax-adjusted"
        
        try:
            reform_ss = reform_model.compute_steady_state(baseline_ss=baseline_ss)
            
            # Get the final residuals to see how close it got
            x0_list = []
            for var_name in reform_model.endogenous_vars_solve:
                val = getattr(reform_ss, var_name)
                if var_name in reform_model.log_vars_indices:
                    val = np.log(val) if val > 1e-9 else np.log(1e-9)
                x0_list.append(val)
            
            final_residuals = reform_model.get_equations_for_steady_state(np.array(x0_list))
            max_residual = np.max(np.abs(final_residuals))
            
            print(f"{tau_c:<10.3f} {change_pp:<8.1f} {strategy:<20} {'SUCCESS':<15} {max_residual:<15.6e}")
            
        except Exception as e:
            print(f"{tau_c:<10.3f} {change_pp:<8.1f} {strategy:<20} {'FAILED':<15} {str(e)[:30]:<15}")

def test_initial_guess_comparison():
    """Compare different initial guess strategies specifically for 2pp"""
    print(f"\n{'='*70}")
    print("INITIAL GUESS STRATEGY COMPARISON FOR 2PP")
    print(f"{'='*70}")
    
    # Setup 2pp consumption tax
    params = ModelParameters.from_json('config/parameters.json')
    baseline_model = DSGEModel(params)
    baseline_ss = baseline_model.compute_steady_state()
    
    reform_params = ModelParameters.from_json('config/parameters.json')
    reform_params.tau_c = 0.12  # 2pp increase
    reform_model = DSGEModel(reform_params)
    
    strategies = [
        ("Baseline values", "baseline"),
        ("Tax-adjusted", "tax_adjusted"),
        ("Interpolation to 1pp", "interp_1pp"),
        ("Interpolation to 3pp", "interp_3pp"),
        ("Conservative tax-adjusted", "conservative"),
    ]
    
    print(f"Testing different initial guess strategies for 2pp consumption tax:")
    print(f"{'Strategy':<25} {'Result':<15} {'Max Residual':<15} {'Notes':<30}")
    print("-" * 85)
    
    for strategy_name, strategy in strategies:
        try:
            if strategy == "baseline":
                initial_guess = {var: getattr(baseline_ss, var) 
                               for var in reform_model.endogenous_vars_solve}
                
            elif strategy == "tax_adjusted":
                initial_guess = reform_model._compute_tax_adjusted_initial_guess(baseline_ss)
                
            elif strategy == "interp_1pp":
                # Interpolate from baseline toward 1pp case
                params_1pp = ModelParameters.from_json('config/parameters.json')
                params_1pp.tau_c = 0.11
                model_1pp = DSGEModel(params_1pp)
                ss_1pp = model_1pp.compute_steady_state(baseline_ss=baseline_ss)
                
                # 50% between baseline and 1pp case
                initial_guess = {}
                for var in reform_model.endogenous_vars_solve:
                    baseline_val = getattr(baseline_ss, var)
                    target_val = getattr(ss_1pp, var)
                    initial_guess[var] = baseline_val + 0.5 * (target_val - baseline_val)
                    
            elif strategy == "interp_3pp":
                # Interpolate from baseline toward 3pp case
                params_3pp = ModelParameters.from_json('config/parameters.json')
                params_3pp.tau_c = 0.13
                model_3pp = DSGEModel(params_3pp)
                ss_3pp = model_3pp.compute_steady_state(baseline_ss=baseline_ss)
                
                # 66% between baseline and 3pp case (2pp out of 3pp)
                initial_guess = {}
                for var in reform_model.endogenous_vars_solve:
                    baseline_val = getattr(baseline_ss, var)
                    target_val = getattr(ss_3pp, var)
                    initial_guess[var] = baseline_val + (2/3) * (target_val - baseline_val)
                    
            elif strategy == "conservative":
                # Very conservative tax-adjusted approach
                initial_guess = {}
                consumption_tax_ratio = (1 + 0.10) / (1 + 0.12)
                for var in reform_model.endogenous_vars_solve:
                    baseline_val = getattr(baseline_ss, var)
                    if var == 'C':
                        initial_guess[var] = baseline_val * (consumption_tax_ratio ** 0.2)  # Small adjustment
                    elif var == 'Lambda':
                        initial_guess[var] = baseline_val / (consumption_tax_ratio ** 0.2)
                    else:
                        initial_guess[var] = baseline_val
            
            # Try to solve with this initial guess
            reform_ss = reform_model.compute_steady_state(initial_guess_dict=initial_guess)
            
            # Get final residuals
            x0_list = []
            for var_name in reform_model.endogenous_vars_solve:
                val = getattr(reform_ss, var_name)
                if var_name in reform_model.log_vars_indices:
                    val = np.log(val) if val > 1e-9 else np.log(1e-9)
                x0_list.append(val)
            
            final_residuals = reform_model.get_equations_for_steady_state(np.array(x0_list))
            max_residual = np.max(np.abs(final_residuals))
            
            print(f"{strategy_name:<25} {'SUCCESS':<15} {max_residual:<15.6e} {'Converged':<30}")
            
        except Exception as e:
            error_msg = str(e)[:30]
            print(f"{strategy_name:<25} {'FAILED':<15} {'N/A':<15} {error_msg:<30}")

def test_manual_solver_settings():
    """Test 2pp with manual solver settings"""
    print(f"\n{'='*70}")
    print("MANUAL SOLVER SETTINGS TEST FOR 2PP")
    print(f"{'='*70}")
    
    # Setup
    params = ModelParameters.from_json('config/parameters.json')
    baseline_model = DSGEModel(params)
    baseline_ss = baseline_model.compute_steady_state()
    
    reform_params = ModelParameters.from_json('config/parameters.json')
    reform_params.tau_c = 0.12
    reform_model = DSGEModel(reform_params)
    
    # Get initial guess (baseline strategy)
    initial_guess = {var: getattr(baseline_ss, var) 
                    for var in reform_model.endogenous_vars_solve}
    
    x0_list = []
    for var_name in reform_model.endogenous_vars_solve:
        val = initial_guess.get(var_name, getattr(baseline_ss, var_name))
        if var_name in reform_model.log_vars_indices:
            val = np.log(val) if val > 1e-9 else np.log(1e-9)
        x0_list.append(val)
    x0 = np.array(x0_list)
    
    # Test different solver configurations
    from scipy.optimize import root
    
    configs = [
        ("hybr, tol=1e-4, iter=1000", {'method': 'hybr', 'options': {'xtol': 1e-4, 'maxfev': 1000}}),
        ("hybr, tol=1e-5, iter=2000", {'method': 'hybr', 'options': {'xtol': 1e-5, 'maxfev': 2000}}),
        ("hybr, tol=1e-6, iter=5000", {'method': 'hybr', 'options': {'xtol': 1e-6, 'maxfev': 5000}}),
        ("hybr, tol=1e-6, iter=10000", {'method': 'hybr', 'options': {'xtol': 1e-6, 'maxfev': 10000}}),
        ("lm, tol=1e-6, iter=3000", {'method': 'lm', 'options': {'xtol': 1e-6, 'maxiter': 3000}}),
        ("broyden1, tol=1e-6, iter=2000", {'method': 'broyden1', 'options': {'xtol': 1e-6, 'maxiter': 2000}}),
    ]
    
    print(f"Testing different solver configurations for 2pp consumption tax:")
    print(f"{'Configuration':<30} {'Success':<10} {'Max Residual':<15} {'Notes':<20}")
    print("-" * 75)
    
    for config_name, config in configs:
        try:
            result = root(reform_model.get_equations_for_steady_state, x0, **config)
            
            final_residuals = reform_model.get_equations_for_steady_state(result.x)
            max_residual = np.max(np.abs(final_residuals))
            
            success = "✅" if result.success else ("🟡" if max_residual < 0.05 else "❌")
            notes = "Converged" if result.success else ("Acceptable" if max_residual < 0.05 else "Failed")
            
            print(f"{config_name:<30} {success:<10} {max_residual:<15.6e} {notes:<20}")
            
        except Exception as e:
            print(f"{config_name:<30} {'💥':<10} {'Exception':<15} {str(e)[:20]:<20}")

def main():
    """Main analysis routine"""
    analyze_2pp_failure()
    test_initial_guess_comparison()
    test_manual_solver_settings()

if __name__ == "__main__":
    main()