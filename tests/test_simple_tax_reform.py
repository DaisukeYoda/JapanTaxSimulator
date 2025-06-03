"""
Simple test to verify the tax reform steady state computation works
without the complex linearization step
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dsge_model import DSGEModel, ModelParameters
from src.tax_simulator import TaxReform

def test_direct_steady_state():
    """Test direct steady state computation for tax reforms"""
    print("=" * 60)
    print("Direct Steady State Test for Tax Reforms")
    print("=" * 60)
    
    # Load baseline
    params = ModelParameters.from_json('config/parameters.json')
    baseline_model = DSGEModel(params)
    baseline_ss = baseline_model.compute_steady_state()
    print(f"✓ Baseline steady state computed")
    print(f"  C: {baseline_ss.C:.3f}, Y: {baseline_ss.Y:.3f}, G: {baseline_ss.G:.3f}")
    
    # Test various tax reforms directly
    test_cases = [
        {"name": "1pp consumption tax", "tau_c": 0.11},
        {"name": "2pp consumption tax", "tau_c": 0.12},
        {"name": "5pp consumption tax", "tau_c": 0.15},
        {"name": "2pp labor tax", "tau_l": 0.22},
        {"name": "Mixed reform", "tau_c": 0.11, "tau_l": 0.22},
    ]
    
    for case in test_cases:
        print(f"\nTesting {case['name']}:")
        
        # Create reform parameters
        reform_params = ModelParameters.from_json('config/parameters.json')
        for key, value in case.items():
            if key != 'name' and hasattr(reform_params, key):
                setattr(reform_params, key, value)
                print(f"  {key}: {getattr(params, key)} -> {value}")
        
        try:
            # Create model and compute steady state with baseline as guide
            reform_model = DSGEModel(reform_params)
            reform_ss = reform_model.compute_steady_state(baseline_ss=baseline_ss)
            
            print(f"  ✓ Converged")
            print(f"    C: {baseline_ss.C:.3f} -> {reform_ss.C:.3f} ({((reform_ss.C/baseline_ss.C-1)*100):+.1f}%)")
            print(f"    Y: {baseline_ss.Y:.3f} -> {reform_ss.Y:.3f} ({((reform_ss.Y/baseline_ss.Y-1)*100):+.1f}%)")
            print(f"    G: {baseline_ss.G:.3f} -> {reform_ss.G:.3f} ({((reform_ss.G/baseline_ss.G-1)*100):+.1f}%)")
            
            # Check if results are economically reasonable
            reasonable = True
            if abs(reform_ss.C/baseline_ss.C - 1) > 0.5:  # >50% change
                print(f"    Warning: Large consumption change")
                reasonable = False
            if abs(reform_ss.Y/baseline_ss.Y - 1) > 0.3:  # >30% change
                print(f"    Warning: Large output change")
                reasonable = False
                
            if reasonable:
                print(f"    ✓ Results appear economically reasonable")
            
        except Exception as e:
            print(f"  ✗ Failed: {str(e)[:60]}...")

def main():
    test_direct_steady_state()

if __name__ == "__main__":
    main()