"""
Detailed analysis of why specific tax reforms fail while others succeed
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dsge_model import DSGEModel, ModelParameters

def analyze_equation_residuals_for_reforms():
    """Analyze equation residuals for failing vs succeeding reforms"""
    print("=" * 70)
    print("EQUATION RESIDUAL ANALYSIS FOR TAX REFORMS")
    print("=" * 70)
    
    # Get baseline
    params = ModelParameters.from_json('config/parameters.json')
    baseline_model = DSGEModel(params)
    baseline_ss = baseline_model.compute_steady_state()
    
    reforms = [
        {"name": "1pp consumption (FAILS)", "tau_c": 0.11, "expected_fail": True},
        {"name": "2pp consumption (WORKS)", "tau_c": 0.12, "expected_fail": False},
        {"name": "5pp consumption (FAILS)", "tau_c": 0.15, "expected_fail": True},
        {"name": "2pp labor (FAILS)", "tau_l": 0.22, "expected_fail": True},
    ]
    
    for reform in reforms:
        print(f"\n{'='*50}")
        print(f"ANALYZING: {reform['name']}")
        print(f"{'='*50}")
        
        # Create reform parameters
        reform_params = ModelParameters.from_json('config/parameters.json')
        for key, value in reform.items():
            if key not in ['name', 'expected_fail'] and hasattr(reform_params, key):
                setattr(reform_params, key, value)
        
        # Try to solve with detailed residual tracking
        reform_model = DSGEModel(reform_params)
        
        try:
            from scipy.optimize import root
            
            # Get initial guess from tax-adjusted method
            initial_guess_dict = reform_model._compute_tax_adjusted_initial_guess(baseline_ss)
            
            # Set up solver input
            x0_list = []
            for var_name in reform_model.endogenous_vars_solve:
                val = initial_guess_dict.get(var_name, getattr(baseline_ss, var_name))
                if var_name in reform_model.log_vars_indices:
                    val = np.log(val) if val > 1e-9 else np.log(1e-9)
                x0_list.append(val)
            x0 = np.array(x0_list)
            
            print(f"Initial guess quality:")
            initial_residuals = reform_model.get_equations_for_steady_state(x0)
            max_initial_residual = np.max(np.abs(initial_residuals))
            print(f"  Max initial residual: {max_initial_residual:.6e}")
            
            # Show largest initial residuals
            eq_names = [
                "Consumption Euler", "Labor supply", "Intertemporal", "Production",
                "Wage", "Capital rental", "Inflation target", "Markup",
                "Government budget", "Fiscal rule", "Taylor rule", "Fisher",
                "Investment", "Asset pricing", "Profit", "Imports", "Exports",
                "NFA target", "Goods market"
            ]
            
            large_residuals = []
            for i, (name, res) in enumerate(zip(eq_names, initial_residuals)):
                if abs(res) > 1e-3:
                    large_residuals.append((name, res))
            
            print(f"  Large initial residuals (>1e-3):")
            for name, res in large_residuals[:5]:  # Top 5
                print(f"    {name:20s}: {res:12.6e}")
            
            # Try to solve
            result = root(reform_model.get_equations_for_steady_state, x0, 
                         method='hybr', options={'xtol': 1e-6, 'maxfev': 5000*(len(x0)+1)})
            
            if result.success:
                print(f"  ✅ CONVERGED")
                final_residuals = reform_model.get_equations_for_steady_state(result.x)
                max_final_residual = np.max(np.abs(final_residuals))
                print(f"  Final max residual: {max_final_residual:.6e}")
            else:
                final_residuals = reform_model.get_equations_for_steady_state(result.x)
                max_final_residual = np.max(np.abs(final_residuals))
                print(f"  ❌ FAILED")
                print(f"  Final max residual: {max_final_residual:.6e}")
                print(f"  Solver message: {result.message}")
                
                # Show worst final residuals
                print(f"  Worst final residuals:")
                worst_indices = np.argsort(np.abs(final_residuals))[-5:]
                for idx in reversed(worst_indices):
                    print(f"    {eq_names[idx]:20s}: {final_residuals[idx]:12.6e}")
                
        except Exception as e:
            print(f"  💥 EXCEPTION: {e}")

def test_convergence_sensitivity():
    """Test sensitivity to convergence criteria"""
    print(f"\n{'='*70}")
    print("CONVERGENCE CRITERIA SENSITIVITY ANALYSIS")
    print(f"{'='*70}")
    
    # Test the problematic 1pp consumption tax with different tolerances
    params = ModelParameters.from_json('config/parameters.json')
    baseline_model = DSGEModel(params)
    baseline_ss = baseline_model.compute_steady_state()
    
    reform_params = ModelParameters.from_json('config/parameters.json')
    reform_params.tau_c = 0.11  # 1pp increase
    reform_model = DSGEModel(reform_params)
    
    tolerances = [1e-4, 1e-5, 1e-6, 1e-7]
    max_iterations = [1000, 2000, 5000, 10000]
    
    print(f"Testing 1pp consumption tax with different solver settings:")
    
    for tol in tolerances:
        for max_iter in max_iterations:
            try:
                from scipy.optimize import root
                
                # Get initial guess
                initial_guess_dict = reform_model._compute_tax_adjusted_initial_guess(baseline_ss)
                x0_list = []
                for var_name in reform_model.endogenous_vars_solve:
                    val = initial_guess_dict.get(var_name, getattr(baseline_ss, var_name))
                    if var_name in reform_model.log_vars_indices:
                        val = np.log(val) if val > 1e-9 else np.log(1e-9)
                    x0_list.append(val)
                x0 = np.array(x0_list)
                
                result = root(reform_model.get_equations_for_steady_state, x0,
                            method='hybr', options={'xtol': tol, 'maxfev': max_iter})
                
                final_residuals = reform_model.get_equations_for_steady_state(result.x)
                max_residual = np.max(np.abs(final_residuals))
                
                status = "✅" if result.success else "❌"
                if max_residual < 0.05:  # Acceptable residual
                    status = "🟡" if not result.success else "✅"
                
                print(f"  tol={tol:.0e}, iter={max_iter:5d}: {status} residual={max_residual:.3e}")
                
                if max_residual < 0.01:  # Found good solution
                    print(f"    🎯 Good solution found!")
                    return tol, max_iter
                
            except Exception as e:
                print(f"  tol={tol:.0e}, iter={max_iter:5d}: 💥 Exception")
    
    return None, None

def test_improved_initial_guess():
    """Test improved initial guess strategies for failing cases"""
    print(f"\n{'='*70}")
    print("IMPROVED INITIAL GUESS STRATEGIES")  
    print(f"{'='*70}")
    
    params = ModelParameters.from_json('config/parameters.json')
    baseline_model = DSGEModel(params)
    baseline_ss = baseline_model.compute_steady_state()
    
    # Test 1pp consumption tax with different initial guess strategies
    reform_params = ModelParameters.from_json('config/parameters.json')
    reform_params.tau_c = 0.11
    reform_model = DSGEModel(reform_params)
    
    strategies = [
        ("Baseline values", "baseline"),
        ("Tax-adjusted", "tax_adjusted"),
        ("Conservative adjustment", "conservative"),
        ("Aggressive adjustment", "aggressive"),
        ("Working case interpolation", "interpolation")
    ]
    
    for strategy_name, strategy in strategies:
        print(f"\nTesting strategy: {strategy_name}")
        
        try:
            if strategy == "baseline":
                # Use baseline steady state values directly
                initial_guess = {var: getattr(baseline_ss, var) 
                               for var in reform_model.endogenous_vars_solve}
            
            elif strategy == "tax_adjusted":
                # Use our current tax-adjusted method
                initial_guess = reform_model._compute_tax_adjusted_initial_guess(baseline_ss)
                
            elif strategy == "conservative":
                # More conservative adjustments (smaller changes)
                initial_guess = {}
                consumption_tax_ratio = (1 + 0.10) / (1 + 0.11)  # 10% -> 11%
                for var in reform_model.endogenous_vars_solve:
                    baseline_val = getattr(baseline_ss, var)
                    if var == 'C':
                        initial_guess[var] = baseline_val * (consumption_tax_ratio ** 0.1)  # Smaller elasticity
                    elif var == 'Lambda':
                        initial_guess[var] = baseline_val / (consumption_tax_ratio ** 0.1)
                    else:
                        initial_guess[var] = baseline_val * (1 + 0.001 * np.random.normal())  # Tiny random
                        
            elif strategy == "aggressive":
                # More aggressive adjustments
                initial_guess = {}
                consumption_tax_ratio = (1 + 0.10) / (1 + 0.11)
                for var in reform_model.endogenous_vars_solve:
                    baseline_val = getattr(baseline_ss, var)
                    if var == 'C':
                        initial_guess[var] = baseline_val * (consumption_tax_ratio ** 1.0)  # Larger elasticity
                    elif var == 'Lambda':
                        initial_guess[var] = baseline_val / (consumption_tax_ratio ** 1.0)
                    elif var == 'L':
                        initial_guess[var] = baseline_val * 0.98  # Reduce labor
                    elif var == 'Y':
                        initial_guess[var] = baseline_val * 0.99  # Reduce output
                    else:
                        initial_guess[var] = baseline_val
                        
            elif strategy == "interpolation":
                # Interpolate between baseline and successful 2pp case
                success_params = ModelParameters.from_json('config/parameters.json')
                success_params.tau_c = 0.12
                success_model = DSGEModel(success_params)
                success_ss = success_model.compute_steady_state(baseline_ss=baseline_ss)
                
                # Interpolate: 50% toward successful case
                alpha = 0.5
                initial_guess = {}
                for var in reform_model.endogenous_vars_solve:
                    baseline_val = getattr(baseline_ss, var)
                    success_val = getattr(success_ss, var)
                    initial_guess[var] = baseline_val + alpha * (success_val - baseline_val)
            
            # Test this initial guess
            reform_ss = reform_model.compute_steady_state(initial_guess_dict=initial_guess)
            print(f"  ✅ SUCCESS with {strategy_name}")
            return strategy_name
            
        except Exception as e:
            print(f"  ❌ FAILED with {strategy_name}: {str(e)[:60]}...")
    
    return None

def main():
    """Main analysis routine"""
    
    # Analysis 1: Equation residuals
    analyze_equation_residuals_for_reforms()
    
    # Analysis 2: Convergence sensitivity
    best_tol, best_iter = test_convergence_sensitivity()
    
    # Analysis 3: Initial guess strategies
    best_strategy = test_improved_initial_guess()
    
    print(f"\n{'='*70}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*70}")
    
    if best_tol:
        print(f"✅ Better solver settings found: tol={best_tol:.0e}, iter={best_iter}")
    else:
        print(f"❌ No better solver settings found")
        
    if best_strategy:
        print(f"✅ Better initial guess strategy found: {best_strategy}")
    else:
        print(f"❌ No better initial guess strategy found")

if __name__ == "__main__":
    main()
