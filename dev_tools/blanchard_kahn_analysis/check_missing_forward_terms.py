#!/usr/bin/env python3
"""
Check which equations should have forward-looking terms but don't
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dsge_model import load_model
from linearization_improved import ImprovedLinearizedDSGE

def check_missing_forward_terms():
    """Check for missing forward-looking terms"""
    print("Checking for missing forward-looking terms...")
    
    # Load model
    model = load_model('config/parameters.json')
    steady_state = model.compute_steady_state()
    
    # Create linearization
    lin_model = ImprovedLinearizedDSGE(model, steady_state)
    
    # Get all equations
    equations = model.get_model_equations()
    
    print(f"\n=== All Model Equations ===")
    print(f"Total equations: {len(equations)}")
    
    # Check each equation for forward-looking variables
    forward_count = 0
    for i, eq in enumerate(equations):
        symbols = eq.free_symbols
        forward_symbols = [s for s in symbols if str(s).endswith('_tp1')]
        
        if forward_symbols:
            forward_count += 1
            print(f"\nEquation {i+1}: Has {len(forward_symbols)} forward-looking terms")
            print(f"  Forward vars: {[str(s) for s in forward_symbols]}")
            print(f"  Equation: {eq}")
    
    print(f"\n=== Summary ===")
    print(f"Equations with forward-looking terms: {forward_count} out of {len(equations)}")
    
    # Check variable classification
    print(f"\n=== Variable Classification ===")
    var_info = lin_model.variable_info
    print(f"Forward-looking variables identified: {var_info['forward_looking']}")
    
    # Cross-check: which forward-looking variables appear in equations
    all_forward_vars = set()
    for eq in equations:
        symbols = eq.free_symbols
        for s in symbols:
            if str(s).endswith('_tp1'):
                base_name = str(s)[:-4]  # Remove _tp1
                all_forward_vars.add(base_name)
    
    print(f"\nActual forward-looking variables in equations: {sorted(list(all_forward_vars))}")
    
    # Check if any important variables are missing forward-looking behavior
    important_vars = ['Y', 'I', 'K', 'B_real', 'NX', 'EX', 'IM', 'w', 'mc', 'profit']
    print(f"\n=== Checking important variables ===")
    for var in important_vars:
        if var in all_forward_vars:
            print(f"✓ {var}: Has forward-looking behavior")
        else:
            print(f"✗ {var}: No forward-looking behavior")

if __name__ == "__main__":
    check_missing_forward_terms()