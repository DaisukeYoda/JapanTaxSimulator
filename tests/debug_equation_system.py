"""
Debug script to analyze the steady state equation system and identify 
why it produces unrealistic values
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dsge_model import DSGEModel, ModelParameters, SteadyState

def analyze_equations_step_by_step():
    """Analyze each equation in the steady state system"""
    print("=" * 60)
    print("Steady State Equation System Analysis")
    print("=" * 60)
    
    params = ModelParameters.from_json('config/parameters.json')
    model = DSGEModel(params)
    
    # Get the computed steady state (even if unrealistic)
    ss = model.compute_steady_state()
    
    print("Current steady state values:")
    print(f"  Y: {ss.Y:.6f}")
    print(f"  C: {ss.C:.6f}")
    print(f"  I: {ss.I:.6f}")
    print(f"  G: {ss.G:.6f}")
    print(f"  K: {ss.K:.6f}")
    print(f"  L: {ss.L:.6f}")
    print(f"  w: {ss.w:.6f}")
    print(f"  Lambda: {ss.Lambda:.6f}")
    print(f"  B_real: {ss.B_real:.6f}")
    
    # Manually compute each equation using steady state values
    print(f"\nManual equation evaluation:")
    
    # Extract values
    Y, C, I, G, K, L = ss.Y, ss.C, ss.I, ss.G, ss.K, ss.L
    w, Rk_gross, r_net_real = ss.w, ss.Rk_gross, ss.r_net_real
    pi_gross, i_nominal_gross = ss.pi_gross, ss.i_nominal_gross
    B_real, Lambda, mc, profit = ss.B_real, ss.Lambda, ss.mc, ss.profit
    q_val, b_star, IM, EX = ss.q, ss.b_star, ss.IM, ss.EX
    
    # Calculate intermediate variables
    A_val = C + I + G
    Y_star_val = params.ystar_ss
    by_target_q = params.by_ratio / 4
    tau_l_effective_ss = params.tau_l_ss + params.phi_b * ((B_real / Y) - by_target_q)
    
    print(f"  tau_l_effective_ss: {tau_l_effective_ss:.6f}")
    print(f"  A_val: {A_val:.6f}")
    print(f"  by_target_q: {by_target_q:.6f}")
    print(f"  B_real/Y: {B_real/Y:.6f}")
    
    # Tax revenues
    Tc_val = params.tau_c * C
    Tl_val = tau_l_effective_ss * w * L
    Tk_val = params.tau_k * Rk_gross * K
    Tf_val = params.tau_f * profit
    T_val = Tc_val + Tl_val + Tk_val + Tf_val
    
    print(f"\nTax revenues:")
    print(f"  Tc_val: {Tc_val:.6f}")
    print(f"  Tl_val: {Tl_val:.6f} (tau_l_eff={tau_l_effective_ss:.6f})")
    print(f"  Tk_val: {Tk_val:.6f}")
    print(f"  Tf_val: {Tf_val:.6f}")
    print(f"  T_val: {T_val:.6f}")
    
    # Evaluate each equation
    equations = [
        ("Consumption Euler", (1-params.beta*params.habit)/(C*(1-params.habit)) - Lambda*(1+params.tau_c)),
        ("Labor supply", params.chi*L**(1/params.sigma_l) - Lambda*(1-tau_l_effective_ss)*w/(1+params.tau_c)),
        ("Intertemporal", Lambda - params.beta*Lambda*(1+r_net_real)/pi_gross),
        ("Production", Y - K**params.alpha * L**(1-params.alpha)),
        ("Wage", w - mc*(1-params.alpha)*Y/L),
        ("Capital rental", Rk_gross - mc*params.alpha*Y/K),
        ("Inflation target", pi_gross - params.pi_target),
        ("Markup", mc - (params.epsilon-1)/params.epsilon),
        ("Government budget", G + r_net_real*B_real - T_val),
        ("Fiscal rule", G - params.gy_ratio*Y*(1-params.phi_b*((B_real/Y)-by_target_q))),
        ("Taylor rule", i_nominal_gross - params.pi_target*(pi_gross/params.pi_target)**params.phi_pi),
        ("Fisher equation", (1+r_net_real) - i_nominal_gross/pi_gross),
        ("Investment", I - params.delta*K),
        ("Asset pricing", (1-params.tau_k)*Rk_gross - params.delta - r_net_real),
        ("Profit", profit - (1-mc)*Y),
        ("Imports", IM - params.alpha_m*A_val*q_val**(-params.eta_im)),
        ("Exports", EX - params.alpha_x*(Y_star_val**params.phi_ex)*q_val**params.eta_ex),
        ("NFA target", b_star - params.b_star_target_level),
        ("Goods market", Y - (C+I+G+q_val*b_star*(1-(1/params.beta))))
    ]
    
    print(f"\nEquation residuals:")
    large_residuals = []
    for name, value in equations:
        print(f"  {name:20s}: {value:12.6e}")
        if abs(value) > 1e-3:
            large_residuals.append((name, value))
    
    print(f"\nLarge residuals (>1e-3):")
    for name, value in large_residuals:
        print(f"  {name:20s}: {value:12.6e}")
    
    return ss, params

def check_parameter_consistency():
    """Check if parameters lead to consistent steady state"""
    print("\n" + "=" * 60)
    print("Parameter Consistency Check")
    print("=" * 60)
    
    params = ModelParameters.from_json('config/parameters.json')
    
    print("Target ratios:")
    print(f"  C/Y target: {params.cy_ratio}")
    print(f"  I/Y target: {params.iy_ratio}")
    print(f"  G/Y target: {params.gy_ratio}")
    print(f"  K/Y target (quarterly): {params.ky_ratio/4:.3f}")
    
    # Check if targets sum to 1 (for closed economy approximation)
    total_ratio = params.cy_ratio + params.iy_ratio + params.gy_ratio
    print(f"  Total C+I+G/Y: {total_ratio:.3f}")
    if total_ratio > 1.05:
        print("  ⚠️  Total exceeds 1 - may need NX adjustment")
    
    # Check production function consistency
    print(f"\nProduction function parameters:")
    print(f"  alpha (capital share): {params.alpha}")
    print(f"  delta (depreciation): {params.delta}")
    print(f"  epsilon (elasticity): {params.epsilon}")
    
    # Check if steady state hours are reasonable
    print(f"  L steady state: {params.hours_steady}")
    if params.hours_steady > 0.5:
        print("  ⚠️  Hours seem high (>50% of time endowment)")
    
    # Check tax rates
    print(f"\nTax rates:")
    print(f"  tau_c: {params.tau_c}")
    print(f"  tau_l: {params.tau_l}")
    print(f"  tau_k: {params.tau_k}")
    print(f"  tau_f: {params.tau_f}")
    
    total_tax_wedge = params.tau_c + params.tau_l + params.tau_k
    print(f"  Total tax wedge: {total_tax_wedge:.3f}")
    if total_tax_wedge > 0.8:
        print("  ⚠️  Very high total tax burden")

def test_realistic_steady_state():
    """Test if we can construct a realistic steady state manually"""
    print("\n" + "=" * 60)
    print("Realistic Steady State Construction")
    print("=" * 60)
    
    params = ModelParameters.from_json('config/parameters.json')
    
    # Start with realistic targets
    Y_target = 1.0  # Normalize output
    C_target = params.cy_ratio * Y_target  # 0.6
    I_target = params.iy_ratio * Y_target  # 0.2
    G_target = params.gy_ratio * Y_target  # 0.2
    
    print(f"Target values:")
    print(f"  Y: {Y_target}")
    print(f"  C: {C_target}")
    print(f"  I: {I_target}")
    print(f"  G: {G_target}")
    
    # Derive other values
    K_target = I_target / params.delta  # From I = δK
    L_target = params.hours_steady  # 0.33
    
    # Check production function
    Y_from_production = K_target**params.alpha * L_target**(1-params.alpha)
    print(f"  Y from production: {Y_from_production:.6f}")
    print(f"  Y target: {Y_target:.6f}")
    print(f"  Ratio: {Y_from_production/Y_target:.3f}")
    
    if abs(Y_from_production/Y_target - 1) > 0.1:
        print("  ⚠️  Production function inconsistency!")
        # Adjust Y to match production
        Y_target = Y_from_production
        C_target = params.cy_ratio * Y_target
        I_target = params.iy_ratio * Y_target
        G_target = params.gy_ratio * Y_target
        print(f"  Adjusted Y: {Y_target}")
    
    # Calculate wages and returns
    mc = (params.epsilon - 1) / params.epsilon
    w_target = mc * (1 - params.alpha) * Y_target / L_target
    Rk_gross_target = mc * params.alpha * Y_target / K_target
    
    print(f"  K: {K_target:.6f}")
    print(f"  L: {L_target:.6f}")
    print(f"  w: {w_target:.6f}")
    print(f"  Rk_gross: {Rk_gross_target:.6f}")
    
    # Check if these values make economic sense
    print(f"\nEconomic reasonableness check:")
    print(f"  K/Y (quarterly): {K_target/Y_target:.3f} (target: {params.ky_ratio/4:.3f})")
    print(f"  w*L/Y (labor share): {w_target*L_target/Y_target:.3f} (should ≈ {1-params.alpha:.3f})")
    print(f"  Rk*K/Y (capital share): {Rk_gross_target*K_target/Y_target:.3f} (should ≈ {params.alpha:.3f})")

def main():
    """Main analysis routine"""
    ss, params = analyze_equations_step_by_step()
    check_parameter_consistency()
    test_realistic_steady_state()

if __name__ == "__main__":
    main()
