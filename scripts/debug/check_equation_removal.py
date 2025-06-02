"""
Check what happens to A matrix after equation removal
"""

import sys
import os

# Add project root to path
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.dsge_model import DSGEModel, ModelParameters
from src.linearization_improved import ImprovedLinearizedDSGE
import numpy as np

def check_equation_removal():
    # Load model
    config_path = os.path.join(os.path.dirname(__file__), '../..', 'config', 'parameters.json')
    params = ModelParameters.from_json(config_path)
    model = DSGEModel(params)
    ss = model.compute_steady_state()
    
    linearizer = ImprovedLinearizedDSGE(model, ss)
    
    print("=== EQUATION REMOVAL ANALYSIS ===")
    
    # Build system matrices (which will trigger equation removal)
    linear_system = linearizer.build_system_matrices()
    
    A_after = linear_system.A
    B_after = linear_system.B
    
    print(f"A matrix after removal: {A_after.shape}")
    print(f"A matrix rank after removal: {np.linalg.matrix_rank(A_after)}/{A_after.shape[0]}")
    
    # Check which rows are zero in the final A matrix
    zero_rows_after = []
    nonzero_rows_after = []
    
    for i in range(A_after.shape[0]):
        row_norm = np.linalg.norm(A_after[i, :])
        if row_norm < 1e-12:
            zero_rows_after.append(i)
        else:
            nonzero_rows_after.append((i, row_norm))
    
    print(f"\nZero rows in final A matrix: {len(zero_rows_after)}/{A_after.shape[0]}")
    print(f"Zero row indices: {zero_rows_after}")
    
    print(f"\nNon-zero rows in final A matrix: {len(nonzero_rows_after)}")
    for i, norm in nonzero_rows_after:
        print(f"  Row {i}: norm = {norm:.6e}")
    
    # Check specific forward-looking entries
    print(f"\n=== FORWARD-LOOKING ENTRIES IN FINAL A MATRIX ===")
    
    # Map from original equation indices to kept equation indices
    # From debug output: Would keep equations: [ 2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 19 20 21 22 23 24 25 26 27 28 29]
    # These are 1-indexed, so subtract 1 for 0-indexed
    kept_equations_0indexed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
    
    # Original forward-looking equations (0-indexed): [2, 4, 5, 7, 13, 20]
    forward_eqs_original = [2, 4, 5, 7, 13, 20]
    
    # Map to new indices
    forward_eqs_kept = []
    for original_idx in forward_eqs_original:
        if original_idx in kept_equations_0indexed:
            new_idx = kept_equations_0indexed.index(original_idx)
            forward_eqs_kept.append((original_idx, new_idx))
    
    print(f"Forward-looking equations kept:")
    for orig_idx, new_idx in forward_eqs_kept:
        print(f"  Original equation {orig_idx + 1} -> New row {new_idx}")
        row_norm = np.linalg.norm(A_after[new_idx, :])
        print(f"    A matrix row norm: {row_norm:.6e}")
        
        # Show non-zero entries in this row
        nonzero_cols = np.where(np.abs(A_after[new_idx, :]) > 1e-12)[0]
        if len(nonzero_cols) > 0:
            print(f"    Non-zero columns: {nonzero_cols.tolist()}")
            for col in nonzero_cols:
                val = A_after[new_idx, col]
                var_name = linearizer.endo_vars[col] if col < len(linearizer.endo_vars) else f"var_{col}"
                print(f"      A[{new_idx}, {col}] ({var_name}): {val:.6e}")
        else:
            print(f"    -> ENTIRE ROW IS ZERO!")
    
    # Check if there's an issue with variable ordering
    print(f"\n=== VARIABLE ORDERING CHECK ===")
    print(f"Endogenous variables ({len(linearizer.endo_vars)}):")
    for i, var in enumerate(linearizer.endo_vars):
        print(f"  {i:2d}: {var}")
    
    # Check specific forward-looking variable columns
    forward_vars = ['C', 'Lambda', 'Rk_gross', 'pi_gross', 'q']
    print(f"\nForward-looking variable columns:")
    for var in forward_vars:
        if var in linearizer.endo_vars:
            col_idx = linearizer.endo_vars.index(var)
            col_norm = np.linalg.norm(A_after[:, col_idx])
            print(f"  {var} (col {col_idx}): ||A[:, {col_idx}]|| = {col_norm:.6e}")
            
            nonzero_rows = np.where(np.abs(A_after[:, col_idx]) > 1e-12)[0]
            if len(nonzero_rows) > 0:
                print(f"    Non-zero rows: {nonzero_rows.tolist()}")
            else:
                print(f"    -> ENTIRE COLUMN IS ZERO!")

if __name__ == "__main__":
    check_equation_removal()