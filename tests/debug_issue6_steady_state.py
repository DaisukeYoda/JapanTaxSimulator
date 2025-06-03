"""
Debug script for Issue #6: Tax Reform Simulations Fail at Counterfactual Steady State Computation

This script reproduces the issue and provides detailed diagnostics for the convergence failure
when computing counterfactual steady states with modified tax parameters.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dsge_model import DSGEModel, ModelParameters
from src.tax_simulator import EnhancedTaxSimulator, TaxReform

def test_baseline_convergence():
    """Test baseline steady state convergence"""
    print("=" * 60)
    print("Testing Baseline Steady State Convergence")
    print("=" * 60)
    
    params = ModelParameters.from_json('config/parameters.json')
    model = DSGEModel(params)
    
    print(f"Baseline parameters:")
    print(f"  tau_c: {params.tau_c}")
    print(f"  tau_l: {params.tau_l}")
    print(f"  tau_k: {params.tau_k}")
    print(f"  tau_f: {params.tau_f}")
    
    try:
        ss = model.compute_steady_state()
        print(f"✓ Baseline steady state converged successfully")
        return True
    except Exception as e:
        print(f"✗ Baseline steady state failed: {e}")
        return False

def test_tax_reform_convergence():
    """Test tax reform steady state convergence with detailed diagnostics"""
    print("\n" + "=" * 60)
    print("Testing Tax Reform Steady State Convergence")
    print("=" * 60)
    
    params = ModelParameters.from_json('config/parameters.json')
    model = DSGEModel(params)
    
    # Compute baseline first
    baseline_ss = model.compute_steady_state()
    print(f"Baseline computed successfully")
    
    # Test the problematic reform: 5pp consumption tax increase
    print(f"\nTesting 5pp consumption tax increase...")
    reform = TaxReform('test_5pp', tau_c=0.15)  # 10% -> 15%
    
    try:
        simulator = EnhancedTaxSimulator(model)
        print(f"✓ Tax simulator created successfully")
        
        # Try to simulate reform (this will compute counterfactual steady state)
        result = simulator.simulate_reform(reform, periods=20)
        print(f"✓ Tax reform simulation completed")
        print(f"  Final residual: Not directly accessible")
        return True
        
    except Exception as e:
        print(f"✗ Tax reform simulation failed: {e}")
        return False

def test_smaller_tax_changes():
    """Test smaller tax changes to find convergence boundary"""
    print("\n" + "=" * 60)
    print("Testing Smaller Tax Changes")
    print("=" * 60)
    
    params = ModelParameters.from_json('config/parameters.json')
    model = DSGEModel(params)
    baseline_ss = model.compute_steady_state()
    
    # Test different tax change magnitudes
    tax_changes = [0.01, 0.02, 0.03, 0.04, 0.05]  # 1pp to 5pp
    
    for change in tax_changes:
        new_tau_c = params.tau_c + change
        print(f"\nTesting {change*100:.0f}pp increase (tau_c: {new_tau_c:.3f})")
        
        reform = TaxReform(f'test_{change*100:.0f}pp', tau_c=new_tau_c)
        
        try:
            simulator = EnhancedTaxSimulator(model)
            result = simulator.simulate_reform(reform, periods=20)
            print(f"  ✓ Converged")
        except Exception as e:
            print(f"  ✗ Failed")
            print(f"    Error: {str(e)[:100]}...")

def analyze_equation_residuals():
    """Analyze individual equation residuals for failed case"""
    print("\n" + "=" * 60)
    print("Analyzing Individual Equation Residuals")
    print("=" * 60)
    
    params = ModelParameters.from_json('config/parameters.json')
    model = DSGEModel(params)
    baseline_ss = model.compute_steady_state()
    
    # Create reform that fails
    reform = TaxReform('analysis', tau_c=0.15)
    simulator = EnhancedTaxSimulator(model)
    
    # Modify model parameters for reform
    model.params.tau_c = reform.tau_c
    
    try:
        # Try to compute with detailed output
        from scipy.optimize import fsolve
        
        def equations_wrapper(vars):
            return model.get_equations_for_steady_state(vars)
        
        # Use baseline as initial guess
        initial_guess = np.array([getattr(baseline_ss, var) for var in model.variable_names])
        
        # Get residuals without solving
        residuals = equations_wrapper(initial_guess)
        print(f"Initial residuals with baseline guess:")
        
        for i, (var, res) in enumerate(zip(model.variable_names, residuals)):
            if abs(res) > 1e-3:  # Only show large residuals
                print(f"  {var:15s}: {res:12.6e}")
                
    except Exception as e:
        print(f"Analysis failed: {e}")

def main():
    """Main diagnostic routine"""
    print("Issue #6 Diagnostic Script")
    print("Tax Reform Counterfactual Steady State Convergence")
    
    # Test 1: Baseline convergence
    baseline_ok = test_baseline_convergence()
    
    if not baseline_ok:
        print("Baseline failed - cannot proceed with reform tests")
        return
    
    # Test 2: Problematic reform
    reform_ok = test_tax_reform_convergence()
    
    # Test 3: Smaller changes
    test_smaller_tax_changes()
    
    # Test 4: Equation analysis
    analyze_equation_residuals()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Baseline convergence: {'✓' if baseline_ok else '✗'}")
    print(f"5pp reform convergence: {'✓' if reform_ok else '✗'}")

if __name__ == "__main__":
    main()
