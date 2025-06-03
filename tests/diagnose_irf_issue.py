#!/usr/bin/env python3
"""
Diagnose Issue #5: Zero Impulse Responses
This test helps identify why IRFs are zero or incorrect
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dsge_model import DSGEModel, ModelParameters
from src.linearization_improved import ImprovedLinearizedDSGE

def diagnose_irf_issue():
    print("=== Diagnosing IRF Issue #5 ===\n")
    
    # Load model
    params = ModelParameters.from_json('config/parameters.json')
    model = DSGEModel(params)
    ss = model.compute_steady_state()
    
    print("1. Steady State Values:")
    print(f"   Y = {ss.Y:.4f}")
    print(f"   A_tfp = {ss.A_tfp:.4f}")
    print(f"   K = {ss.K:.4f}")
    print(f"   L = {ss.L:.4f}")
    
    # Linearize
    linearizer = ImprovedLinearizedDSGE(model, ss)
    linear_system = linearizer.build_system_matrices()
    
    print("\n2. System Dimensions:")
    print(f"   A matrix: {linear_system.A.shape}")
    print(f"   B matrix: {linear_system.B.shape}")
    print(f"   C matrix: {linear_system.C.shape}")
    print(f"   Rank A: {np.linalg.matrix_rank(linear_system.A)}")
    print(f"   Rank B: {np.linalg.matrix_rank(linear_system.B)}")
    
    print("\n3. Variable Order:")
    print(f"   Endogenous vars: {linearizer.endo_vars[:10]}...")
    print(f"   Exogenous vars: {linearizer.exo_vars}")
    
    # Check where TFP appears
    a_tfp_idx = linearizer.endo_vars.index('A_tfp')
    y_idx = linearizer.endo_vars.index('Y')
    print(f"\n4. Variable Indices:")
    print(f"   A_tfp index: {a_tfp_idx}")
    print(f"   Y index: {y_idx}")
    
    # Check C matrix (shock loading)
    eps_a_idx = linearizer.exo_vars.index('eps_a')
    print(f"\n5. Shock Loading (C matrix):")
    print(f"   eps_a index: {eps_a_idx}")
    
    # Find non-zero entries in eps_a column
    eps_a_col = linear_system.C[:, eps_a_idx]
    non_zero_rows = np.where(np.abs(eps_a_col) > 1e-10)[0]
    
    print(f"\n   Non-zero entries in eps_a column:")
    for row in non_zero_rows:
        var_name = linearizer.endo_vars[row] if row < len(linearizer.endo_vars) else f"row_{row}"
        print(f"   Row {row} ({var_name}): {eps_a_col[row]:.6f}")
    
    # Check if TFP equation is correctly captured
    print(f"\n6. TFP Equation Analysis:")
    
    # Look at A_tfp row in matrices
    print(f"   A matrix row {a_tfp_idx} (A_tfp forward):")
    print(f"   {linear_system.A[a_tfp_idx, :]}")
    
    print(f"\n   B matrix row {a_tfp_idx} (A_tfp current):")
    print(f"   {linear_system.B[a_tfp_idx, :]}")
    
    print(f"\n   C matrix row {a_tfp_idx} (A_tfp shocks):")
    print(f"   {linear_system.C[a_tfp_idx, :]}")
    
    # Check production function equation
    print(f"\n7. Production Function Analysis:")
    print(f"   B matrix row {y_idx} (Y equation):")
    non_zero_cols = np.where(np.abs(linear_system.B[y_idx, :]) > 1e-10)[0]
    for col in non_zero_cols:
        var_name = linearizer.endo_vars[col] if col < len(linearizer.endo_vars) else f"col_{col}"
        print(f"     Col {col} ({var_name}): {linear_system.B[y_idx, col]:.6f}")
    
    # Try Klein solution
    print(f"\n8. Klein Solution:")
    try:
        P, Q = linearizer.solve_klein()
        print(f"   P matrix shape: {P.shape}")
        print(f"   Q matrix shape: {Q.shape}")
        
        # Check eigenvalues
        full_system = np.block([
            [linearizer.linear_system.A, np.zeros_like(linearizer.linear_system.B)],
            [np.zeros_like(linearizer.linear_system.A), np.eye(linearizer.linear_system.A.shape[0])]
        ])
        companion = np.block([
            [-linearizer.linear_system.B, -linearizer.linear_system.A],
            [np.eye(linearizer.linear_system.A.shape[0]), np.zeros_like(linearizer.linear_system.A)]
        ])
        
        eigenvalues = np.linalg.eigvals(companion) / np.linalg.eigvals(full_system)
        eigenvalues = eigenvalues[np.isfinite(eigenvalues)]
        n_explosive = np.sum(np.abs(eigenvalues) > 1.0)
        
        print(f"   Number of explosive eigenvalues: {n_explosive}")
        print(f"   Number of forward-looking variables: {linearizer.n_f}")
        
    except Exception as e:
        print(f"   Klein solution failed: {e}")
    
    # Simple IRF test
    print(f"\n9. Simple IRF Test:")
    try:
        # Direct shock to A_tfp
        shock_vector = np.zeros(len(linearizer.exo_vars))
        shock_vector[eps_a_idx] = 0.01  # 1% shock
        
        # One-period response using C matrix
        initial_response = linear_system.C @ shock_vector
        
        print(f"   Initial response to 1% TFP shock:")
        significant_responses = []
        for i, response in enumerate(initial_response):
            if abs(response) > 1e-6:
                var_name = linearizer.endo_vars[i] if i < len(linearizer.endo_vars) else f"var_{i}"
                print(f"     {var_name}: {response:.6f}")
                significant_responses.append((var_name, response))
        
        if len(significant_responses) == 0:
            print(f"     WARNING: No significant responses found!")
            
    except Exception as e:
        print(f"   IRF test failed: {e}")
    
    print("\n=== Diagnosis Complete ===")

if __name__ == "__main__":
    diagnose_irf_issue()