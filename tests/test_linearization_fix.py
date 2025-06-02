#!/usr/bin/env python3
"""
Test script for improved linearization
"""

import numpy as np
import os
from src.dsge_model import load_model
from src.linearization_improved import ImprovedLinearizedDSGE

def test_improved_linearization():
    """Test the improved symbolic linearization"""
    print("Testing improved linearization...")
    
    # Load model
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'parameters.json')
    model = load_model(config_path)
    steady_state = model.compute_steady_state()
    
    # Create improved linearization
    lin_model = ImprovedLinearizedDSGE(model, steady_state)
    
    # Print variable classification
    print("\n=== Variable Classification ===")
    print(f"Endogenous variables ({len(lin_model.endo_vars)}): {lin_model.endo_vars}")
    print(f"Exogenous variables ({len(lin_model.exo_vars)}): {lin_model.exo_vars}")
    print(f"Predetermined variables ({len(lin_model.state_vars)}): {lin_model.state_vars}")
    print(f"Jump variables ({len(lin_model.control_vars)}): {lin_model.control_vars}")
    print(f"Forward-looking variables: {lin_model.variable_info['forward_looking']}")
    
    # Build system matrices
    print("\n=== Building System Matrices ===")
    try:
        linear_system = lin_model.build_system_matrices()
        
        print(f"System built successfully!")
        print(f"A matrix shape: {linear_system.A.shape}")
        print(f"B matrix shape: {linear_system.B.shape}")
        print(f"C matrix shape: {linear_system.C.shape}")
        
        # Check matrix properties
        A_rank = np.linalg.matrix_rank(linear_system.A)
        B_rank = np.linalg.matrix_rank(linear_system.B)
        
        print(f"\nMatrix properties:")
        print(f"A matrix rank: {A_rank} (out of {min(linear_system.A.shape)})")
        print(f"B matrix rank: {B_rank} (out of {min(linear_system.B.shape)})")
        
        # Check for forward-looking terms
        A_nonzero = np.count_nonzero(linear_system.A)
        print(f"Non-zero elements in A matrix: {A_nonzero}")
        
        if A_nonzero > 0:
            print("✓ Forward-looking terms detected in A matrix")
        else:
            print("⚠ No forward-looking terms in A matrix")
            
        # Test solution
        print("\n=== Testing Solution ===")
        try:
            P, Q = lin_model.solve_klein()
            print("✓ Klein solution completed")
            print(f"Policy matrix P shape: {P.shape}")
            print(f"Transition matrix Q shape: {Q.shape}")
            
        except Exception as e:
            print(f"✗ Klein solution failed: {e}")
            
    except Exception as e:
        print(f"Failed to build system matrices: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_improved_linearization()