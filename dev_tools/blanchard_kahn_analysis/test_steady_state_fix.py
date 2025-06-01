#!/usr/bin/env python3
"""
Test the steady state computation with the original model
to ensure we can still solve it before adding complexity
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Temporarily revert to simpler model for steady state
from src.dsge_model import DSGEModel, ModelParameters

def test_steady_state():
    """Test steady state computation"""
    
    print("Testing steady state computation...")
    
    # Initialize model
    model = DSGEModel()
    model.load_parameters('config/parameters.json')
    
    # Try with better initial guess
    initial_guess = {
        'C': 0.6,  # Consumption around 60% of GDP
        'I': 0.2,  # Investment around 20% of GDP
        'Y': 1.0,  # Normalize output to 1
        'K': 7.4,  # Capital stock (quarterly GDP * 1.85 * 4)
        'L': 0.33, # Hours worked
        'w': 0.67, # Real wage (labor share of output)
        'G': 0.2,  # Government spending 20% of GDP
        'B_real': 0.5, # Government debt (quarterly)
        'Lambda': 1.0,  # Marginal utility
        'pi_gross': 1.005, # Quarterly inflation target
        'i_nominal_gross': 1.015, # Nominal interest rate
        'r_net_real': 0.01, # Real interest rate
        'mc': 0.833, # Marginal cost
        'Rk_gross': 0.04, # Return on capital
        'profit': 0.167, # Profits
        'A_dom': 1.0, # Domestic absorption
        'IM': 0.15, # Imports
        'EX': 0.15, # Exports
    }
    
    try:
        ss = model.compute_steady_state(initial_guess_dict=initial_guess)
        print("✓ Steady state computed successfully")
        
        # Check steady state
        errors = model.check_steady_state(ss)
        print("\nSteady state values:")
        print(f"  Y = {ss.Y:.4f}")
        print(f"  C = {ss.C:.4f} (C/Y = {ss.C/ss.Y:.3f})")
        print(f"  I = {ss.I:.4f} (I/Y = {ss.I/ss.Y:.3f})")
        print(f"  G = {ss.G:.4f} (G/Y = {ss.G/ss.Y:.3f})")
        print(f"  L = {ss.L:.4f}")
        print(f"  K = {ss.K:.4f} (K/Y_annual = {ss.K/(4*ss.Y):.3f})")
        
        print("\nKey errors:")
        for key, error in errors.items():
            if abs(error) > 1e-6 and not key.startswith('Eq_Resid_'):
                print(f"  {key}: {error:.6f}")
                
        return True
    except Exception as e:
        print(f"✗ Failed to compute steady state: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_steady_state()
    sys.exit(0 if success else 1)