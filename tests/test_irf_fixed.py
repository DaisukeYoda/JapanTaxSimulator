#!/usr/bin/env python3
"""
Test IRF with fixed linearization
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from src.dsge_model import load_model
from src.linearization_improved import ImprovedLinearizedDSGE

def test_fixed_irf():
    """Test IRF computation with the fixed linearization"""
    print("Testing IRF with fixed linearization...")
    
    # Load model and create linearization
    model = load_model(os.path.join(os.path.dirname(__file__), '..', 'config', 'parameters.json'))
    steady_state = model.compute_steady_state()
    lin_model = ImprovedLinearizedDSGE(model, steady_state)
    
    # Solve the model
    print("Solving model...")
    try:
        P, Q = lin_model.solve_klein()
        print("✓ Model solved successfully")
        
        # Test IRF computation for government spending shock
        print("\nComputing government spending shock IRF...")
        variables = ['Y', 'C', 'I', 'G']
        irf_data = lin_model.compute_impulse_response(
            shock_type='gov_spending',
            shock_size=1.0,
            periods=20,
            variables=variables
        )
        
        print("✓ IRF computed successfully")
        print("\nIRF Summary (first 5 periods):")
        print(irf_data.head())
        
        # Create a simple plot
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()
        
        for i, var in enumerate(variables):
            if var in irf_data.columns:
                axes[i].plot(irf_data.index, irf_data[var], linewidth=2)
                axes[i].set_title(f'{var}')
                axes[i].grid(True, alpha=0.3)
                axes[i].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        plt.suptitle('Government Spending Shock IRF (Fixed Linearization)', fontsize=14)
        plt.tight_layout()
        plt.savefig('results/irf_fixed_linearization.png', dpi=150, bbox_inches='tight')
        print("✓ IRF plot saved to results/irf_fixed_linearization.png")
        
        # Check if results are reasonable
        gov_impact = irf_data['G'].iloc[0] if 'G' in irf_data.columns else 0
        output_impact = irf_data['Y'].iloc[0] if 'Y' in irf_data.columns else 0
        
        print(f"\nImpact analysis:")
        print(f"Government spending impact: {gov_impact:.3f}%")
        print(f"Output impact: {output_impact:.3f}%")
        
        if abs(gov_impact) > 0.1 and abs(output_impact) > 0.05:
            print("✓ IRF responses look reasonable")
        else:
            print("⚠ IRF responses may be too small")
            
    except Exception as e:
        print(f"✗ Failed to compute IRF: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_irf()