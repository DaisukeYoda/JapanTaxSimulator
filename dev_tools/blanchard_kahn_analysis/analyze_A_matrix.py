#!/usr/bin/env python3
"""
Analyze the A matrix to understand why its rank is low
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dsge_model import load_model
from linearization_improved import ImprovedLinearizedDSGE

def analyze_A_matrix():
    """Analyze the A matrix structure"""
    print("Analyzing A matrix structure...")
    
    # Load model
    model = load_model('config/parameters.json')
    steady_state = model.compute_steady_state()
    
    # Create linearization
    lin_model = ImprovedLinearizedDSGE(model, steady_state)
    linear_system = lin_model.build_system_matrices()
    
    A = linear_system.A
    B = linear_system.B
    
    print(f"\n=== A Matrix Analysis ===")
    print(f"Shape: {A.shape}")
    print(f"Rank: {np.linalg.matrix_rank(A)}")
    print(f"Non-zero elements: {np.count_nonzero(A)}")
    
    # Find which rows and columns have non-zero elements
    row_nonzero = np.any(A != 0, axis=1)
    col_nonzero = np.any(A != 0, axis=0)
    
    print(f"\nRows with non-zero elements: {np.sum(row_nonzero)} out of {A.shape[0]}")
    print(f"Columns with non-zero elements: {np.sum(col_nonzero)} out of {A.shape[1]}")
    
    # Show which variables have forward-looking terms
    print("\n=== Variables with forward-looking terms (columns) ===")
    for i, var in enumerate(lin_model.endo_vars):
        if col_nonzero[i]:
            print(f"{var}: column {i}")
    
    # Show which equations have forward-looking terms (rows)
    print("\n=== Equations with forward-looking terms (rows) ===")
    for i in range(A.shape[0]):
        if row_nonzero[i]:
            print(f"Equation {i+1}")
    
    # Look at the actual non-zero values
    print("\n=== Non-zero elements in A ===")
    rows, cols = np.where(A != 0)
    for r, c in zip(rows, cols):
        print(f"A[{r},{c}] = {A[r,c]:.6f} (Eq {r+1}, var {lin_model.endo_vars[c]})")
    
    # Check if the issue is with specific equations
    print("\n=== Checking model equations ===")
    print(f"Number of equations: {len(lin_model.equations)}")
    
    # Sample a few equations to see their structure
    for i in [2, 4, 5, 7, 13, 20]:  # Equations with forward-looking terms
        if i < len(lin_model.equations):
            eq = lin_model.equations[i]
            print(f"\nEquation {i+1}: {eq}")
            symbols = eq.free_symbols
            forward_symbols = [s for s in symbols if str(s).endswith('_tp1')]
            print(f"Forward-looking symbols: {[str(s) for s in forward_symbols]}")

if __name__ == "__main__":
    analyze_A_matrix()