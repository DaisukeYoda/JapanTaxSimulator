#!/usr/bin/env python3
"""
Trace equation filtering to understand the mapping
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dsge_model import DSGEModel, ModelParameters
from src.linearization_improved import ImprovedLinearizedDSGE

def trace_equation_filtering():
    print("=== Tracing Equation Filtering ===\n")
    
    # Load model
    params = ModelParameters.from_json('config/parameters.json')
    model = DSGEModel(params)
    ss = model.compute_steady_state()
    
    # Get original equations
    equations = model.get_model_equations()
    
    # Find equation 9 (TFP equation with eps_a)
    tfp_eq_idx = 9
    tfp_eq = equations[tfp_eq_idx]
    print(f"Original equation {tfp_eq_idx}: {tfp_eq}")
    print(f"Contains eps_a: {'eps_a' in str(tfp_eq)}")
    
    # Build linearizer
    linearizer = ImprovedLinearizedDSGE(model, ss)
    
    # From our debug, we know the kept equations are:
    # [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    # In 0-indexed: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
    
    kept_equations_1indexed = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    kept_equations_0indexed = [i-1 for i in kept_equations_1indexed]
    
    print(f"\nKept equations (0-indexed): {kept_equations_0indexed}")
    
    # Find where equation 9 maps to in the final system
    if 9 in kept_equations_0indexed:
        final_row = kept_equations_0indexed.index(9)
        print(f"Equation 9 (TFP with eps_a) maps to final row: {final_row}")
        
        # This should be where eps_a appears in the C matrix
        print(f"Expected: eps_a should affect row {final_row} in final C matrix")
        print(f"Actual: eps_a affects row 8 in final C matrix")
        
        if final_row != 8:
            print(f"✗ MISMATCH: Expected row {final_row}, got row 8")
        else:
            print(f"✓ MATCH: Correct row mapping")
            
    # Check variable ordering
    print(f"\nVariable ordering:")
    for i, var in enumerate(linearizer.endo_vars[:15]):
        print(f"  {i}: {var}")
        
    # The issue might be that equation 9 defines/affects a different variable than A_tfp
    # Let's check which variable equation 9 is "supposed" to define
    
    print(f"\nChecking TFP equation content...")
    tfp_eq_str = str(tfp_eq)
    print(f"Equation 9: {tfp_eq_str}")
    
    # Extract the main variable (left-hand side)
    if hasattr(tfp_eq, 'lhs'):
        lhs_str = str(tfp_eq.lhs)
        print(f"LHS: {lhs_str}")
        
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    trace_equation_filtering()