"""
Debug script to analyze the steady state values and understand 
why tax reforms produce unrealistic results
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dsge_model import DSGEModel, ModelParameters

def analyze_baseline_steady_state():
    """Analyze baseline steady state values in detail"""
    print("=" * 60)
    print("Baseline Steady State Analysis")
    print("=" * 60)
    
    params = ModelParameters.from_json('config/parameters.json')
    model = DSGEModel(params)
    ss = model.compute_steady_state()
    
    print("Parameters:")
    print(f"  tau_c: {params.tau_c}")
    print(f"  tau_l: {params.tau_l}")
    print(f"  tau_k: {params.tau_k}")
    print(f"  tau_f: {params.tau_f}")
    print(f"  cy_ratio: {params.cy_ratio}")
    print(f"  iy_ratio: {params.iy_ratio}")
    print(f"  gy_ratio: {params.gy_ratio}")
    
    print(f"\nKey Steady State Values:")
    print(f"  Y: {ss.Y:.6f}")
    print(f"  C: {ss.C:.6f}")
    print(f"  I: {ss.I:.6f}")
    print(f"  G: {ss.G:.6f}")
    print(f"  K: {ss.K:.6f}")
    print(f"  L: {ss.L:.6f}")
    print(f"  w: {ss.w:.6f}")
    print(f"  Lambda: {ss.Lambda:.6f}")
    print(f"  B_real: {ss.B_real:.6f}")
    
    print(f"\nRatios:")
    print(f"  C/Y: {ss.C/ss.Y:.3f} (target: {params.cy_ratio})")
    print(f"  I/Y: {ss.I/ss.Y:.3f} (target: {params.iy_ratio})")
    print(f"  G/Y: {ss.G/ss.Y:.3f} (target: {params.gy_ratio})")
    print(f"  K/Y: {ss.K/ss.Y:.3f} (target: {params.ky_ratio/4:.3f})")  # Quarterly
    
    print(f"\nTax Revenues:")
    print(f"  Tc: {ss.Tc:.6f}")
    print(f"  Tl: {ss.Tl:.6f}")
    print(f"  Tk: {ss.Tk:.6f}")
    print(f"  Tf: {ss.Tf:.6f}")
    print(f"  T_total: {ss.T_total_revenue:.6f}")
    print(f"  T_total/Y: {ss.T_total_revenue/ss.Y:.3f}")
    
    # Check accounting identity
    lhs = ss.Y
    rhs = ss.C + ss.I + ss.G + ss.NX
    print(f"\nAccounting Check:")
    print(f"  Y = {lhs:.6f}")
    print(f"  C+I+G+NX = {rhs:.6f}")
    print(f"  Difference: {abs(lhs-rhs):.6e}")
    
    # Check government budget
    gov_spending = ss.G + ss.r_net_real * ss.B_real
    gov_revenue = ss.T_total_revenue
    print(f"\nGovernment Budget:")
    print(f"  Spending (G + rB): {gov_spending:.6f}")
    print(f"  Revenue (T): {gov_revenue:.6f}")
    print(f"  Surplus: {gov_revenue - gov_spending:.6f}")
    
    return ss, params

def test_consumption_tax_effect():
    """Test the effect of consumption tax changes step by step"""
    print("\n" + "=" * 60)
    print("Consumption Tax Effect Analysis")
    print("=" * 60)
    
    # Get baseline
    baseline_params = ModelParameters.from_json('config/parameters.json')
    baseline_model = DSGEModel(baseline_params)
    baseline_ss = baseline_model.compute_steady_state()
    
    # Test small consumption tax increase
    reform_params = ModelParameters.from_json('config/parameters.json')
    reform_params.tau_c = 0.11  # 10% -> 11%
    
    print(f"Tax change: tau_c {baseline_params.tau_c} -> {reform_params.tau_c}")
    
    try:
        reform_model = DSGEModel(reform_params)
        reform_ss = reform_model.compute_steady_state(baseline_ss=baseline_ss)
        
        print(f"\nResults:")
        print(f"  Y: {baseline_ss.Y:.6f} -> {reform_ss.Y:.6f} ({((reform_ss.Y/baseline_ss.Y-1)*100):+.2f}%)")
        print(f"  C: {baseline_ss.C:.6f} -> {reform_ss.C:.6f} ({((reform_ss.C/baseline_ss.C-1)*100):+.2f}%)")
        print(f"  I: {baseline_ss.I:.6f} -> {reform_ss.I:.6f} ({((reform_ss.I/baseline_ss.I-1)*100):+.2f}%)")
        print(f"  G: {baseline_ss.G:.6f} -> {reform_ss.G:.6f} ({((reform_ss.G/baseline_ss.G-1)*100):+.2f}%)")
        print(f"  L: {baseline_ss.L:.6f} -> {reform_ss.L:.6f} ({((reform_ss.L/baseline_ss.L-1)*100):+.2f}%)")
        print(f"  w: {baseline_ss.w:.6f} -> {reform_ss.w:.6f} ({((reform_ss.w/baseline_ss.w-1)*100):+.2f}%)")
        print(f"  Lambda: {baseline_ss.Lambda:.6f} -> {reform_ss.Lambda:.6f} ({((reform_ss.Lambda/baseline_ss.Lambda-1)*100):+.2f}%)")
        
        print(f"\nTax Revenues:")
        print(f"  Tc: {baseline_ss.Tc:.6f} -> {reform_ss.Tc:.6f} ({((reform_ss.Tc/baseline_ss.Tc-1)*100):+.2f}%)")
        print(f"  Total: {baseline_ss.T_total_revenue:.6f} -> {reform_ss.T_total_revenue:.6f} ({((reform_ss.T_total_revenue/baseline_ss.T_total_revenue-1)*100):+.2f}%)")
        
        # Check ratios
        print(f"\nRatios (Reform):")
        print(f"  C/Y: {reform_ss.C/reform_ss.Y:.3f}")
        print(f"  I/Y: {reform_ss.I/reform_ss.Y:.3f}")
        print(f"  G/Y: {reform_ss.G/reform_ss.Y:.3f}")
        
        # Check if the huge changes make sense
        print(f"\nDiagnostics:")
        
        # Expected consumption tax effect
        expected_consumption_effect = baseline_params.tau_c / reform_params.tau_c
        print(f"  Expected consumption adjustment ratio: {expected_consumption_effect:.3f}")
        print(f"  Actual consumption ratio: {reform_ss.C/baseline_ss.C:.3f}")
        
        if reform_ss.C/baseline_ss.C > 2.0:
            print(f"  ⚠️  Consumption increase is unrealistically large!")
            
        if reform_ss.G/baseline_ss.G > 2.0:
            print(f"  ⚠️  Government spending increase is unrealistically large!")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")

def check_equation_residuals():
    """Check individual equation residuals for problematic cases"""
    print("\n" + "=" * 60)
    print("Equation Residuals Analysis")
    print("=" * 60)
    
    # Test the problematic case
    params = ModelParameters.from_json('config/parameters.json')
    params.tau_c = 0.15  # Large consumption tax
    
    model = DSGEModel(params)
    
    try:
        # Try to solve and examine residuals
        from scipy.optimize import root
        
        # Get default initial guess
        baseline_params = ModelParameters.from_json('config/parameters.json')
        baseline_model = DSGEModel(baseline_params)
        baseline_ss = baseline_model.compute_steady_state()
        
        ss = model.compute_steady_state(baseline_ss=baseline_ss)
        
        # Get final residuals
        x0_list = []
        for var_name in model.endogenous_vars_solve:
            val = getattr(ss, var_name)
            if var_name in model.log_vars_indices:
                val = np.log(val) if val > 1e-9 else np.log(1e-9)
            x0_list.append(val)
        
        residuals = model.get_equations_for_steady_state(np.array(x0_list))
        
        print(f"Final residuals:")
        for i, (var, res) in enumerate(zip(model.endogenous_vars_solve, residuals)):
            if abs(res) > 1e-6:
                print(f"  {var:15s}: {res:12.6e}")
        
        print(f"Max residual: {np.max(np.abs(residuals)):.6e}")
        
    except Exception as e:
        print(f"Analysis failed: {e}")

def main():
    """Main analysis routine"""
    baseline_ss, baseline_params = analyze_baseline_steady_state()
    test_consumption_tax_effect()
    check_equation_residuals()

if __name__ == "__main__":
    main()