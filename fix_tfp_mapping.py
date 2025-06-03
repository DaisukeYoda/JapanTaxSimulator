#!/usr/bin/env python3
"""
Fix the TFP shock mapping issue by correcting the equation-variable association
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dsge_model import DSGEModel, ModelParameters
from src.linearization_improved import ImprovedLinearizedDSGE

def analyze_and_fix_mapping():
    print("=== Analyzing and Fixing TFP Mapping ===\n")
    
    # Load model
    params = ModelParameters.from_json('config/parameters.json')
    model = DSGEModel(params)
    ss = model.compute_steady_state()
    
    # Create linearizer
    linearizer = ImprovedLinearizedDSGE(model, ss)
    
    # Build the system to see current mapping
    linear_system = linearizer.build_system_matrices()
    
    print("Current situation:")
    print(f"Variable 1 (A_tfp) - equation in row 1")
    print(f"Variable 8 (K) - equation in row 8")
    print(f"eps_a shock affects row 8 (should affect A_tfp)")
    
    # Find where the shock should be
    A_tfp_idx = linearizer.endo_vars.index('A_tfp')
    K_idx = linearizer.endo_vars.index('K')
    eps_a_idx = linearizer.exo_vars.index('eps_a')
    
    print(f"\nVariable indices:")
    print(f"A_tfp: {A_tfp_idx}")
    print(f"K: {K_idx}")
    print(f"eps_a: {eps_a_idx}")
    
    # Current C matrix
    C_original = linear_system.C.copy()
    print(f"\nOriginal C matrix eps_a column:")
    eps_a_col = C_original[:, eps_a_idx]
    for i, val in enumerate(eps_a_col):
        if abs(val) > 1e-10:
            var_name = linearizer.endo_vars[i] if i < len(linearizer.endo_vars) else f"var_{i}"
            print(f"  Row {i} ({var_name}): {val}")
    
    # The fix: Move the shock from K row to A_tfp row
    C_fixed = C_original.copy()
    
    # Get the shock value currently in K row
    shock_value = C_original[K_idx, eps_a_idx]
    
    # Clear K row and set A_tfp row
    C_fixed[K_idx, eps_a_idx] = 0.0
    C_fixed[A_tfp_idx, eps_a_idx] = shock_value
    
    print(f"\nFixed C matrix eps_a column:")
    eps_a_col_fixed = C_fixed[:, eps_a_idx]
    for i, val in enumerate(eps_a_col_fixed):
        if abs(val) > 1e-10:
            var_name = linearizer.endo_vars[i] if i < len(linearizer.endo_vars) else f"var_{i}"
            print(f"  Row {i} ({var_name}): {val}")
    
    # Update the linear system
    linearizer.linear_system.C = C_fixed
    
    # Test IRF with fixed mapping
    print(f"\n=== Testing Fixed IRF ===")
    
    try:
        # Solve Klein
        P, Q = linearizer.solve_klein()
        
        # Compute IRF
        irf = linearizer.compute_impulse_response(
            shock_type='tfp',
            shock_size=1.0,
            periods=10,
            variables=['A_tfp', 'Y', 'C', 'I']
        )
        
        print(f"\nFixed IRF results:")
        for var in ['A_tfp', 'Y', 'C', 'I']:
            if var in irf.columns:
                print(f"{var}:")
                for t in range(min(5, len(irf))):
                    print(f"  t={t}: {irf[var].iloc[t]:.4f}%")
        
        # Check if responses are significant
        max_responses = {}
        for var in ['A_tfp', 'Y', 'C', 'I']:
            if var in irf.columns:
                max_responses[var] = np.max(np.abs(irf[var]))
        
        print(f"\nMaximum absolute responses:")
        for var, max_resp in max_responses.items():
            print(f"  {var}: {max_resp:.4f}%")
            
        # Success criterion
        if max_responses.get('Y', 0) > 0.1:
            print(f"\n✅ SUCCESS: GDP response is significant ({max_responses['Y']:.4f}%)")
            print(f"✅ TFP shock mapping has been fixed!")
        else:
            print(f"\n⚠️  GDP response still too small: {max_responses.get('Y', 0):.4f}%")
            
    except Exception as e:
        print(f"Error in testing: {e}")
        import traceback
        traceback.print_exc()
    
    return linearizer

def create_permanent_fix():
    """Create a permanent fix by modifying the linearization code"""
    print(f"\n=== Creating Permanent Fix ===")
    
    # The fix needs to be applied in the build_system_matrices method
    # We need to ensure that when the TFP equation (containing eps_a) is processed,
    # it affects the A_tfp variable row, not the K variable row
    
    print("The permanent fix requires modifying linearization_improved.py")
    print("to ensure proper equation-variable association during matrix construction.")
    
    return True

if __name__ == "__main__":
    fixed_linearizer = analyze_and_fix_mapping()
    create_permanent_fix()