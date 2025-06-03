"""
Test script to validate the steady state computation fix for tax parameter changes
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dsge_model import DSGEModel, ModelParameters, SteadyState
from src.tax_simulator import EnhancedTaxSimulator, TaxReform

def test_improved_initial_guess():
    """Test improved initial guess generation for tax reforms"""
    print("=" * 60)
    print("Testing Improved Initial Guess for Tax Reforms")
    print("=" * 60)
    
    # Load baseline parameters
    params = ModelParameters.from_json('config/parameters.json')
    baseline_model = DSGEModel(params)
    baseline_ss = baseline_model.compute_steady_state()
    print(f"✓ Baseline steady state computed")
    
    # Create model with higher consumption tax
    reform_params = ModelParameters.from_json('config/parameters.json')
    reform_params.tau_c = 0.15  # 10% -> 15%
    
    print(f"\nTesting tax reform (tau_c: {params.tau_c} -> {reform_params.tau_c})")
    
    # Try computation with improved initial guess
    reform_model = DSGEModel(reform_params)
    
    # Create smart initial guess based on baseline steady state
    initial_guess = {}
    for var in reform_model.endogenous_vars_solve:
        baseline_val = getattr(baseline_ss, var)
        
        # Adjust initial guess based on expected effects of tax change
        if var == 'C':
            # Higher consumption tax should reduce consumption
            tax_effect = (1 + params.tau_c) / (1 + reform_params.tau_c)
            initial_guess[var] = baseline_val * tax_effect
        elif var == 'Lambda':
            # Marginal utility should increase with higher tax (lower consumption)
            tax_effect = (1 + reform_params.tau_c) / (1 + params.tau_c)
            initial_guess[var] = baseline_val * tax_effect
        elif var == 'G':
            # Government spending should adjust for higher tax revenue
            # Assume some of the extra revenue goes to spending
            extra_revenue = (reform_params.tau_c - params.tau_c) * baseline_ss.C
            initial_guess[var] = baseline_val + 0.5 * extra_revenue
        elif var == 'B_real':
            # Government debt should decrease with higher revenue
            extra_revenue = (reform_params.tau_c - params.tau_c) * baseline_ss.C
            initial_guess[var] = max(baseline_val - 2 * extra_revenue, 0.5 * baseline_val)
        else:
            # For other variables, use baseline values
            initial_guess[var] = baseline_val
    
    try:
        reform_ss = reform_model.compute_steady_state(initial_guess_dict=initial_guess)
        print(f"✓ Reform steady state computed successfully with smart initial guess")
        
        # Show key changes
        print(f"\nKey Changes:")
        print(f"  C: {baseline_ss.C:.3f} -> {reform_ss.C:.3f} ({((reform_ss.C/baseline_ss.C-1)*100):+.1f}%)")
        print(f"  G: {baseline_ss.G:.3f} -> {reform_ss.G:.3f} ({((reform_ss.G/baseline_ss.G-1)*100):+.1f}%)")
        print(f"  B_real: {baseline_ss.B_real:.3f} -> {reform_ss.B_real:.3f} ({((reform_ss.B_real/baseline_ss.B_real-1)*100):+.1f}%)")
        print(f"  Lambda: {baseline_ss.Lambda:.3f} -> {reform_ss.Lambda:.3f} ({((reform_ss.Lambda/baseline_ss.Lambda-1)*100):+.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"✗ Reform steady state failed even with smart initial guess: {e}")
        return False

def test_robust_parameter_bounds():
    """Test steady state computation with parameter bound checking"""
    print("\n" + "=" * 60)
    print("Testing Robust Parameter Bounds")
    print("=" * 60)
    
    params = ModelParameters.from_json('config/parameters.json')
    
    # Test various tax rate combinations
    test_cases = [
        {"name": "High consumption tax", "tau_c": 0.20},
        {"name": "High labor tax", "tau_l": 0.40},
        {"name": "High capital tax", "tau_k": 0.40},
        {"name": "High corporate tax", "tau_f": 0.45},
        {"name": "Moderate increases", "tau_c": 0.12, "tau_l": 0.25},
    ]
    
    for case in test_cases:
        print(f"\nTesting {case['name']}:")
        
        # Create test parameters
        test_params = ModelParameters.from_json('config/parameters.json')
        for key, value in case.items():
            if key != 'name' and hasattr(test_params, key):
                setattr(test_params, key, value)
                print(f"  {key}: {getattr(params, key)} -> {value}")
        
        # Check parameter validity
        valid = True
        if test_params.tau_k >= 0.95:
            print(f"    Warning: tau_k too high ({test_params.tau_k})")
            valid = False
        if test_params.tau_l >= 0.95:
            print(f"    Warning: tau_l too high ({test_params.tau_l})")
            valid = False
        if test_params.tau_c >= 0.5:
            print(f"    Warning: tau_c very high ({test_params.tau_c})")
        
        if not valid:
            print(f"    Skipping due to invalid parameters")
            continue
        
        try:
            model = DSGEModel(test_params)
            ss = model.compute_steady_state()
            print(f"    ✓ Converged")
        except Exception as e:
            print(f"    ✗ Failed: {str(e)[:50]}...")

def main():
    """Main test routine"""
    print("Steady State Computation Fix Tests")
    
    # Test 1: Improved initial guess
    smart_guess_works = test_improved_initial_guess()
    
    # Test 2: Parameter bounds
    test_robust_parameter_bounds()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Smart initial guess: {'✓' if smart_guess_works else '✗'}")

if __name__ == "__main__":
    main()
