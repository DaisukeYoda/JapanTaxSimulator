#!/usr/bin/env python3
"""
Debug script for linearization
"""

import numpy as np
import os
import sympy
import sys

# Add project root to path when running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.dsge_model import load_model
from src.linearization_improved import ImprovedLinearizedDSGE

def debug_linearization():
    """Debug the symbolic linearization"""
    print("Debugging symbolic linearization...")
    
    # Load model
    config_path = os.path.join(os.path.dirname(__file__), '../..', 'config', 'parameters.json')
    model = load_model(config_path)
    steady_state = model.compute_steady_state()
    
    # Create linearization
    lin_model = ImprovedLinearizedDSGE(model, steady_state)
    
    # Look at a sample equation
    eq_sample = lin_model.equations[0]
    print(f"\nSample equation: {eq_sample}")
    print(f"LHS: {eq_sample.lhs}")
    print(f"RHS: {eq_sample.rhs}")
    
    # Check symbols in the equation
    all_symbols = eq_sample.free_symbols
    print(f"\nSymbols in sample equation: {[str(s) for s in all_symbols]}")
    
    # Check variable extraction
    print(f"\nEndogenous variables: {lin_model.endo_vars[:10]}...")
    print(f"Exogenous variables: {lin_model.exo_vars}")
    
    # Test symbolic differentiation on a simple case
    print("\n=== Testing Symbolic Differentiation ===")
    
    # Take the first equation and examine it
    expr = eq_sample.lhs - eq_sample.rhs
    print(f"Expression: {expr}")
    
    # Try differentiating with respect to a variable we know should be there
    test_vars = ['C_t', 'Lambda_t', 'tau_l_effective_t']
    for var_name in test_vars:
        if var_name.replace('_t', '') in lin_model.endo_vars:
            var_symbol = sympy.Symbol(var_name)
            try:
                deriv = sympy.diff(expr, var_symbol)
                print(f"d/d{var_name}: {deriv}")
            except Exception as e:
                print(f"Failed to differentiate w.r.t. {var_name}: {e}")

if __name__ == "__main__":
    debug_linearization()