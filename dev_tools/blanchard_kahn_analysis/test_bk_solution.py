#!/usr/bin/env python3
"""
Test the Blanchard-Kahn solution with enhanced forward-looking dynamics
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dsge_model import DSGEModel, ModelParameters
from src.linearization_improved_bk_fix import ImprovedLinearizedDSGE

def test_bk_solution():
    """Test the enhanced BK solution"""
    
    print("=" * 70)
    print("Testing Enhanced Blanchard-Kahn Solution")
    print("=" * 70)
    
    # Initialize model
    print("\n1. Initializing model...")
    model = DSGEModel(ModelParameters.from_json('config/parameters.json'))
    
    # Compute steady state
    print("\n2. Computing steady state...")
    try:
        ss = model.compute_steady_state()
        print("✓ Steady state computed successfully")
        print(f"  Y_ss = {ss.Y:.4f}")
        print(f"  C_ss = {ss.C:.4f}")
        print(f"  I_ss = {ss.I:.4f}")
    except Exception as e:
        print(f"✗ Failed to compute steady state: {e}")
        return False
    
    # Linearize with enhanced forward-looking dynamics
    print("\n3. Linearizing with enhanced forward-looking dynamics...")
    try:
        lin_model = ImprovedLinearizedDSGE(model)
        lin_model.linearize_model()
        print("✓ Enhanced linearization completed")
    except Exception as e:
        print(f"✗ Linearization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check Blanchard-Kahn conditions
    print("\n4. Checking Blanchard-Kahn conditions...")
    n_explosive, n_forward, satisfied = lin_model.check_blanchard_kahn()
    
    print(f"  Number of explosive eigenvalues: {n_explosive}")
    print(f"  Number of forward-looking variables: {n_forward}")
    print(f"  Forward-looking variables: {', '.join(lin_model.forward_looking_vars)}")
    
    if satisfied:
        print("✓ BLANCHARD-KAHN CONDITIONS SATISFIED!")
        success = True
    else:
        print("✗ Blanchard-Kahn conditions NOT satisfied")
        success = False
        
        # Additional diagnostics
        print("\n5. Additional diagnostics:")
        
        # Check which equations contribute to forward dynamics
        print("\n  Equations with forward-looking terms:")
        forward_eq_count = 0
        for i, info in enumerate(lin_model.equation_info):
            if info.get('forward_vars'):
                forward_eq_count += 1
                desc = info.get('description', f'Equation {i+1}')
                vars_str = ', '.join(info['forward_vars'])
                print(f"    {desc}: {vars_str}")
        
        print(f"\n  Total equations with forward terms: {forward_eq_count}/{len(lin_model.equation_info)}")
        
        # Analyze A matrix structure
        A = lin_model.A_full
        A_rank = np.linalg.matrix_rank(A)
        print(f"\n  A matrix rank: {A_rank}/{A.shape[0]}")
        
        # Find rows with no forward-looking terms
        zero_rows = []
        for i in range(A.shape[0]):
            if np.max(np.abs(A[i, :])) < 1e-10:
                zero_rows.append(i)
        
        if zero_rows:
            print(f"  Equations with no forward terms: {len(zero_rows)}")
    
    return success

def main():
    """Main test function"""
    success = test_bk_solution()
    
    print("\n" + "=" * 70)
    if success:
        print("SUCCESS: Blanchard-Kahn conditions satisfied with enhanced model!")
        print("\nThe solution approach:")
        print("1. Added more forward-looking variables (Y, L, w, mc, r_net_real)")
        print("2. Introduced additional dynamic equations")
        print("3. Enhanced weak forward-looking equations")
    else:
        print("FAILURE: Still working on satisfying Blanchard-Kahn conditions")
        print("\nNext steps to try:")
        print("1. Adjust the forward-looking weights")
        print("2. Add more dynamic structure to static equations")
        print("3. Review parameter calibration")
    print("=" * 70)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())