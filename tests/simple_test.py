"""Simple steady state convergence test"""

import numpy as np
import os
from src.dsge_model import DSGEModel, ModelParameters, SteadyState
from scipy.optimize import fsolve

# Load model
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'parameters.json')
params = ModelParameters.from_json(config_path)
model = DSGEModel(params)

# Try with default initial guess
print("Testing with default initial guess...")
ss = SteadyState()
ss.Y = 1.0
ss.C = 0.6
ss.I = 0.2
ss.K = 10.0
ss.L = 0.33
ss.G = 0.2
ss.B_real = 0.5  # Start with lower debt level
ss.pi_gross = 1.005
ss.r_net_real = 0.0101
ss.i_nominal_gross = 1.0151
ss.w = 2.0
ss.Rk_gross = 0.047
ss.Lambda = 1.0
ss.mc = 0.833
ss.profit = 0.167
ss.q = 1.0
ss.b_star = 0.0
ss.IM = 0.15
ss.EX = 0.15

# Convert to solver format
x0 = []
for var in model.endogenous_vars_solve:
    val = getattr(ss, var)
    if var in model.log_vars_indices:
        val = np.log(val) if val > 1e-9 else np.log(1e-9)
    x0.append(val)
x0 = np.array(x0)

# Check residuals
residuals = model.get_equations_for_steady_state(x0)
print(f"Max residual: {np.max(np.abs(residuals)):.6e}")

# Try with very loose tolerance first
try:
    result = fsolve(model.get_equations_for_steady_state, x0, xtol=1e-4, maxfev=100, full_output=True)
    x_sol, info, ier, msg = result
    print(f"fsolve with loose tolerance: {msg}")
    if ier == 1:
        final_residuals = model.get_equations_for_steady_state(x_sol)
        print(f"Final max residual: {np.max(np.abs(final_residuals)):.6e}")
except Exception as e:
    print(f"Error: {e}")

# Try simplifying the initial guess - make it more consistent
print("\nTrying with simplified, consistent initial guess...")

# Set up very basic initial values based on parameter targets
Y_init = 1.0
C_init = params.cy_ratio * Y_init  # 0.6
I_init = params.iy_ratio * Y_init  # 0.2  
G_init = params.gy_ratio * Y_init  # 0.2
K_init = params.ky_ratio * Y_init / 4  # Convert annual to quarterly: 2.5
L_init = params.hours_steady  # 0.33

# Based on production function: Y = K^alpha * L^(1-alpha)
# Check if this is consistent
Y_check = K_init**params.alpha * L_init**(1-params.alpha)
print(f"Y consistency check: Target={Y_init:.3f}, Implied={Y_check:.3f}")

# Calculate real interest rate from Euler equation
r_init = (1/params.beta) - 1  # 0.0101
Rk_init = (r_init + params.delta) / (1 - params.tau_k)  # Real return to capital

# Calculate wages from production function
mc_init = (params.epsilon - 1) / params.epsilon
w_init = mc_init * (1 - params.alpha) * Y_init / L_init

# Set up new initial guess
x0_simple = []
for var in model.endogenous_vars_solve:
    if var == 'Y': val = Y_init
    elif var == 'C': val = C_init
    elif var == 'I': val = I_init
    elif var == 'K': val = K_init
    elif var == 'L': val = L_init
    elif var == 'G': val = G_init
    elif var == 'B_real': val = (params.by_ratio/4) * Y_init  # 0.5
    elif var == 'r_net_real': val = r_init
    elif var == 'Rk_gross': val = Rk_init
    elif var == 'w': val = w_init
    elif var == 'pi_gross': val = params.pi_target
    elif var == 'i_nominal_gross': val = params.pi_target * (1 + r_init)
    elif var == 'mc': val = mc_init
    elif var == 'Lambda': val = 1.0
    elif var == 'profit': val = (1 - mc_init) * Y_init
    elif var == 'q': val = 1.0
    elif var == 'b_star': val = 0.0
    elif var == 'IM': val = 0.15
    elif var == 'EX': val = 0.15
    else: val = 1.0
    
    if var in model.log_vars_indices:
        val = np.log(val) if val > 1e-9 else np.log(1e-9)
    x0_simple.append(val)

x0_simple = np.array(x0_simple)

# Check residuals with simplified guess
residuals_simple = model.get_equations_for_steady_state(x0_simple)
print(f"Max residual (simplified): {np.max(np.abs(residuals_simple)):.6e}")

print(f"\nWorst equations (simplified guess):")
worst_idx = np.argsort(np.abs(residuals_simple))[-5:][::-1]
for idx in worst_idx:
    print(f"  Eq[{idx:2d}] ({model.endogenous_vars_solve[idx]:12s}): {residuals_simple[idx]:12.6e}")
