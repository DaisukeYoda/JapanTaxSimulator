"""
Final detailed debug of A matrix structure
"""

import sys
import os

# Add project root to path
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.dsge_model import DSGEModel, ModelParameters
from src.linearization_improved import ImprovedLinearizedDSGE
import numpy as np

def final_matrix_debug():
    # Load model
    config_path = os.path.join(os.path.dirname(__file__), '../..', 'config', 'parameters.json')
    params = ModelParameters.from_json(config_path)
    model = DSGEModel(params)
    ss = model.compute_steady_state()
    
    linearizer = ImprovedLinearizedDSGE(model, ss)
    linear_system = linearizer.build_system_matrices()
    
    A = linear_system.A
    B = linear_system.B
    
    print("=== FINAL A MATRIX STRUCTURE DEBUG ===")
    print(f"A matrix shape: {A.shape}")
    print(f"A matrix rank: {np.linalg.matrix_rank(A)}")
    
    # Show the actual A matrix (first few rows/cols)
    print(f"\nA matrix (first 10x10):")
    print(A[:10, :10])
    
    # Find all non-zero entries in A
    nonzero_row, nonzero_col = np.where(np.abs(A) > 1e-12)
    print(f"\nNon-zero entries in A matrix: {len(nonzero_row)}")
    
    if len(nonzero_row) > 0:
        print(f"Non-zero A matrix entries:")
        for i in range(min(20, len(nonzero_row))):  # Show first 20
            row, col = nonzero_row[i], nonzero_col[i]
            val = A[row, col]
            var_name = linearizer.endo_vars[col] if col < len(linearizer.endo_vars) else f"var_{col}"
            print(f"  A[{row:2d}, {col:2d}] ({var_name:15s}): {val:12.6e}")
        
        if len(nonzero_row) > 20:
            print(f"  ... and {len(nonzero_row) - 20} more entries")
    
    # Check specifically for the zero pattern
    print(f"\nZero analysis:")
    zero_rows = []
    for i in range(A.shape[0]):
        if np.allclose(A[i, :], 0, atol=1e-12):
            zero_rows.append(i)
    
    print(f"Zero rows: {len(zero_rows)}/{A.shape[0]}")
    print(f"Zero row indices: {zero_rows}")
    
    # Check the issue: why is rank only 5?
    print(f"\n=== RANK ANALYSIS ===")
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    print(f"SVD analysis:")
    print(f"Singular values (> 1e-10): {s[s > 1e-10]}")
    print(f"Number of significant singular values: {np.sum(s > 1e-10)}")
    
    # Check if the problem is that A is not square
    print(f"\nA matrix shape analysis:")
    print(f"Shape: {A.shape}")
    print(f"Square matrix? {A.shape[0] == A.shape[1]}")
    
    if A.shape[0] != A.shape[1]:
        print(f"A matrix is not square! This explains the rank issue.")
        print(f"Need {A.shape[1]} equations but only have {A.shape[0]}")
        
        # The issue might be that we removed too many equations
        print(f"\nThis suggests the equation removal process needs revision.")
    
    # Let's also check what happens if we force a square system
    if A.shape[0] < A.shape[1]:
        print(f"\nForcing a square system by adding zero equations...")
        n_missing = A.shape[1] - A.shape[0]
        A_square = np.vstack([A, np.zeros((n_missing, A.shape[1]))])
        B_square = np.vstack([B, np.zeros((n_missing, B.shape[1]))])
        
        print(f"Square A matrix shape: {A_square.shape}")
        print(f"Square A matrix rank: {np.linalg.matrix_rank(A_square)}")
        
        # Check determinant of square version
        try:
            det_A = np.linalg.det(A_square)
            print(f"Determinant of square A: {det_A:.2e}")
        except:
            print("Cannot compute determinant")

if __name__ == "__main__":
    final_matrix_debug()