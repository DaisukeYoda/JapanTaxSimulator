"""Test the initial guess computation"""

from src.dsge_model import ModelParameters, SteadyState
import numpy as np

params = ModelParameters.from_json('config/parameters.json')

# Test the new initial guess logic
ss_defaults = SteadyState()
ss_defaults.tau_l_effective = params.tau_l_ss
ss_defaults.pi_gross = params.pi_target
ss_defaults.r_net_real = (1/params.beta) - 1 
ss_defaults.i_nominal_net = (1 + ss_defaults.r_net_real) * ss_defaults.pi_gross - 1
ss_defaults.i_nominal_gross = 1 + ss_defaults.i_nominal_net
ss_defaults.Rk_gross = (ss_defaults.r_net_real + params.delta) / (1 - params.tau_k if (1-params.tau_k) > 0.1 else 1.0)
ss_defaults.mc = (params.epsilon - 1) / params.epsilon
ss_defaults.Y = 1.0

# Apply the new logic
ss_defaults.K = params.ky_ratio * ss_defaults.Y / 4  # Convert annual to quarterly
ss_defaults.L = params.hours_steady

print(f"Initial values:")
print(f"Y: {ss_defaults.Y}")
print(f"K: {ss_defaults.K}")
print(f"L: {ss_defaults.L}")
print(f"ky_ratio (annual): {params.ky_ratio}")
print(f"K/Y (quarterly): {ss_defaults.K/ss_defaults.Y}")

# Check if these K, L values are consistent with production function
Y_implied = ss_defaults.K**params.alpha * ss_defaults.L**(1-params.alpha)
print(f"Y_implied from production function: {Y_implied}")

# If not consistent, adjust L to match production function while keeping K/Y ratio
if abs(Y_implied - ss_defaults.Y) > 1e-6:
    print("Adjusting L to match production function...")
    # Solve: Y = K^alpha * L^(1-alpha) for L
    # L = (Y / K^alpha)^(1/(1-alpha))
    if ss_defaults.K > 1e-9:
        ss_defaults.L = (ss_defaults.Y / (ss_defaults.K**params.alpha))**(1/(1-params.alpha))
        print(f"New L: {ss_defaults.L}")
        # Check consistency
        Y_check = ss_defaults.K**params.alpha * ss_defaults.L**(1-params.alpha)
        print(f"Y consistency check: {Y_check}")

# Check other derived values
ss_defaults.I = params.delta * ss_defaults.K
ss_defaults.G = params.gy_ratio * ss_defaults.Y 
ss_defaults.C = ss_defaults.Y - ss_defaults.I - ss_defaults.G

print(f"\nDerived values:")
print(f"I: {ss_defaults.I}")
print(f"G: {ss_defaults.G}")
print(f"C: {ss_defaults.C}")
print(f"C/Y: {ss_defaults.C/ss_defaults.Y}")
print(f"I/Y: {ss_defaults.I/ss_defaults.Y}")
print(f"G/Y: {ss_defaults.G/ss_defaults.Y}")

# Check if C > 0
if ss_defaults.C <= 0:
    print("WARNING: C <= 0! This will cause major problems.")

# Check ratios against targets
print(f"\nRatio checks:")
print(f"C/Y target: {params.cy_ratio}, actual: {ss_defaults.C/ss_defaults.Y}")
print(f"I/Y target: {params.iy_ratio}, actual: {ss_defaults.I/ss_defaults.Y}")
print(f"G/Y target: {params.gy_ratio}, actual: {ss_defaults.G/ss_defaults.Y}")

# The issue might be that I/Y + G/Y + C/Y doesn't add up correctly
total_ratio = (ss_defaults.C + ss_defaults.I + ss_defaults.G) / ss_defaults.Y
print(f"(C+I+G)/Y: {total_ratio} (should be 1.0)")