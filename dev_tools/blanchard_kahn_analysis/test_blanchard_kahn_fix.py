#!/usr/bin/env python3
"""
Test script to verify Blanchard-Kahn conditions after model improvements
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dsge_model import DSGEModel, ModelParameters
from src.linearization_improved import ImprovedLinearizedDSGE as LinearizedDSGE

def test_blanchard_kahn_conditions():
    """Test if the improved model satisfies Blanchard-Kahn conditions"""
    
    print("=" * 70)
    print("Testing Blanchard-Kahn Conditions with Improved Model")
    print("=" * 70)
    
    # Initialize model
    model = DSGEModel()
    model.load_parameters('config/parameters.json')
    
    # Compute steady state
    print("\n1. Computing steady state...")
    try:
        ss = model.compute_steady_state()
        print("✓ Steady state computed successfully")
        print(f"  Y_ss = {ss.Y:.4f}")
        print(f"  C_ss = {ss.C:.4f}")
        print(f"  I_ss = {ss.I:.4f}")
        print(f"  L_ss = {ss.L:.4f}")
    except Exception as e:
        print(f"✗ Failed to compute steady state: {e}")
        return False
    
    # Linearize model
    print("\n2. Linearizing the model...")
    try:
        lin_model = LinearizedDSGE(model)
        lin_model.linearize_model()
        print("✓ Model linearized successfully")
    except Exception as e:
        print(f"✗ Failed to linearize model: {e}")
        return False
    
    # Get system matrices
    print("\n3. Extracting system matrices...")
    A_full = lin_model.A_full
    B_full = lin_model.B_full
    
    print(f"  A matrix shape: {A_full.shape}")
    print(f"  B matrix shape: {B_full.shape}")
    print(f"  A matrix rank: {np.linalg.matrix_rank(A_full)}")
    print(f"  B matrix rank: {np.linalg.matrix_rank(B_full)}")
    
    # Count forward-looking variables
    forward_vars = lin_model.forward_looking_vars
    n_forward = len(forward_vars)
    print(f"\n4. Forward-looking variables ({n_forward}):")
    for var in forward_vars:
        print(f"  - {var}")
    
    # Check which equations have forward-looking terms
    print("\n5. Equations with forward-looking terms:")
    for i, eq_info in enumerate(lin_model.equation_info):
        if 'forward_vars' in eq_info and eq_info['forward_vars']:
            print(f"  Equation {i+1}: {eq_info.get('description', 'No description')}")
            print(f"    Forward vars: {eq_info['forward_vars']}")
    
    # Solve using Klein method
    print("\n6. Solving with Klein method...")
    try:
        gx, hx = model.klein_solve(A_full, B_full, n_forward)
        print("✓ Klein solution obtained")
    except Exception as e:
        print(f"✗ Klein solution failed: {e}")
        return False
    
    # Analyze eigenvalues more carefully
    print("\n7. Detailed eigenvalue analysis:")
    from scipy.linalg import qz
    AA, BB, alpha, beta, Q, Z = qz(A_full, B_full, output='complex')
    
    with np.errstate(divide='ignore', invalid='ignore'):
        eigenvalues = np.where(np.abs(beta) > 1e-10, 
                             np.abs(alpha/beta), 
                             np.inf)
    
    # Sort eigenvalues
    sorted_eigs = np.sort(eigenvalues)
    n_explosive = np.sum(eigenvalues > 1.0 + 1e-6)
    n_unit_root = np.sum(np.abs(eigenvalues - 1.0) < 1e-6)
    
    print(f"  Total eigenvalues: {len(eigenvalues)}")
    print(f"  Explosive (|λ| > 1): {n_explosive}")
    print(f"  Unit root (|λ| ≈ 1): {n_unit_root}")
    print(f"  Stable (|λ| < 1): {len(eigenvalues) - n_explosive - n_unit_root}")
    
    print("\n  First 10 eigenvalues (sorted):")
    for i, eig in enumerate(sorted_eigs[:10]):
        print(f"    λ{i+1} = {eig:.6f}")
    
    print("\n  Last 10 eigenvalues (sorted):")
    for i, eig in enumerate(sorted_eigs[-10:]):
        print(f"    λ{len(sorted_eigs)-9+i} = {eig:.6f}")
    
    # Check Blanchard-Kahn conditions
    print("\n8. Blanchard-Kahn Condition Check:")
    print(f"  Number of explosive eigenvalues: {n_explosive}")
    print(f"  Number of forward-looking variables: {n_forward}")
    
    if n_explosive == n_forward:
        print("✓ BLANCHARD-KAHN CONDITIONS SATISFIED!")
        print("  The model has a unique stable solution.")
        success = True
    elif n_explosive < n_forward:
        print("✗ BLANCHARD-KAHN CONDITIONS NOT SATISFIED")
        print("  Model is INDETERMINATE (multiple stable solutions)")
        success = False
    else:
        print("✗ BLANCHARD-KAHN CONDITIONS NOT SATISFIED")
        print("  Model has NO STABLE SOLUTION")
        success = False
    
    # Test impulse responses if BK conditions are satisfied
    if success:
        print("\n9. Testing impulse responses...")
        try:
            irfs = model.compute_impulse_responses('eps_a', shock_size=0.01, 
                                                  periods=20, use_linearization=lin_model)
            
            # Check if responses are non-zero
            key_vars = ['Y', 'C', 'I', 'L', 'w', 'pi_gross']
            print("  Peak responses to 1% TFP shock:")
            for var in key_vars:
                if var in irfs:
                    peak = np.max(np.abs(irfs[var]))
                    print(f"    {var}: {peak:.4f}%")
            
            # Save IRF plot
            model.plot_irfs(irfs, 'eps_a', save_path='results/irf_after_bk_fix.png')
            print("✓ Impulse responses computed and saved")
        except Exception as e:
            print(f"✗ Failed to compute impulse responses: {e}")
    
    return success

def main():
    """Main test function"""
    success = test_blanchard_kahn_conditions()
    
    print("\n" + "=" * 70)
    if success:
        print("SUCCESS: Model improvements resolved Blanchard-Kahn issues!")
    else:
        print("FAILURE: Blanchard-Kahn conditions still not satisfied.")
        print("\nPossible next steps:")
        print("1. Review the new forward-looking equations")
        print("2. Check parameter calibration")
        print("3. Consider alternative model specifications")
    print("=" * 70)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())