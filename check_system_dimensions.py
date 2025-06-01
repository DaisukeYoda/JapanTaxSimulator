#!/usr/bin/env python3
"""
Check system dimensions
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dsge_model import load_model
from linearization_improved import ImprovedLinearizedDSGE

def check_dimensions():
    """Check system dimensions"""
    print("Checking system dimensions...")
    
    # Load model
    model = load_model('config/parameters.json')
    steady_state = model.compute_steady_state()
    
    # Create linearization
    lin_model = ImprovedLinearizedDSGE(model, steady_state)
    
    print(f"Number of equations: {len(lin_model.equations)}")
    print(f"Number of endogenous variables: {len(lin_model.endo_vars)}")
    print(f"Number of exogenous variables: {len(lin_model.exo_vars)}")
    
    # List all equations
    print("\n=== Model Equations ===")
    for i, eq in enumerate(lin_model.equations):
        print(f"Eq {i+1}: {eq}")
    
    # Build matrices and check which have non-zero rows
    linear_system = lin_model.build_system_matrices()
    
    print(f"\n=== Matrix Analysis ===")
    print(f"A matrix: {linear_system.A.shape}")
    print(f"B matrix: {linear_system.B.shape}")
    
    # Check which equations are non-trivial
    non_zero_rows_A = np.any(linear_system.A != 0, axis=1)
    non_zero_rows_B = np.any(linear_system.B != 0, axis=1)
    non_zero_rows = non_zero_rows_A | non_zero_rows_B
    
    print(f"Non-zero rows in A: {np.sum(non_zero_rows_A)}")
    print(f"Non-zero rows in B: {np.sum(non_zero_rows_B)}")
    print(f"Total non-trivial equations: {np.sum(non_zero_rows)}")
    
    print(f"\nEquations with non-zero coefficients:")
    for i, is_nonzero in enumerate(non_zero_rows):
        if is_nonzero:
            print(f"Eq {i+1}: {lin_model.equations[i]}")

if __name__ == "__main__":
    check_dimensions()