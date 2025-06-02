"""Debug linearization matrices to identify issues"""

import sys
import os

# Add project root to path when running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.dsge_model import DSGEModel, ModelParameters
from src.linearization_improved import ImprovedLinearizedDSGE
import numpy as np

# Load model and compute steady state
config_path = os.path.join(os.path.dirname(__file__), '../..', 'config', 'parameters.json')
params = ModelParameters.from_json(config_path)
model = DSGEModel(params)
print("Computing steady state...")
ss = model.compute_steady_state()
print(f"✓ Steady state computed: Y={ss.Y:.3f}")

# Create linearizer
linearizer = ImprovedLinearizedDSGE(model, ss)
print(f"\nVariable ordering:")
print(f"Endogenous vars ({len(linearizer.endo_vars)}): {linearizer.endo_vars}")
print(f"State vars ({len(linearizer.state_vars)}): {linearizer.state_vars}")
print(f"Control vars ({len(linearizer.control_vars)}): {linearizer.control_vars}")

# Build system
linear_system = linearizer.build_system_matrices()
A, B, C = linear_system.A, linear_system.B, linear_system.C

print(f"\nMatrix properties:")
print(f"A shape: {A.shape}, B shape: {B.shape}, C shape: {C.shape}")

# Check for issues in A matrix
print(f"\nA matrix analysis:")
print(f"Condition number: {np.linalg.cond(A):.2e}")
print(f"Determinant: {np.linalg.det(A):.2e}")
print(f"Rank: {np.linalg.matrix_rank(A)}/{A.shape[0]}")

# Find zero rows in A
zero_rows = []
for i in range(A.shape[0]):
    if np.allclose(A[i, :], 0, atol=1e-12):
        zero_rows.append(i)
        
if zero_rows:
    print(f"Zero rows in A: {zero_rows}")
    for row in zero_rows:
        print(f"  Equation {row} ({linearizer.endo_vars[row] if row < len(linearizer.endo_vars) else 'unknown'})")

# Find zero columns in A
zero_cols = []
for j in range(A.shape[1]):
    if np.allclose(A[:, j], 0, atol=1e-12):
        zero_cols.append(j)
        
if zero_cols:
    print(f"Zero columns in A: {zero_cols}")
    for col in zero_cols:
        print(f"  Variable {col} ({linearizer.endo_vars[col] if col < len(linearizer.endo_vars) else 'unknown'})")

# Check eigenvalues of A
try:
    eigenvals_A = np.linalg.eigvals(A)
    zero_eigenvals = np.sum(np.abs(eigenvals_A) < 1e-10)
    print(f"Zero eigenvalues in A: {zero_eigenvals}")
except:
    print("Could not compute eigenvalues of A")

# Check B matrix
print(f"\nB matrix analysis:")
print(f"Condition number: {np.linalg.cond(B):.2e}")
print(f"Determinant: {np.linalg.det(B):.2e}")
print(f"Rank: {np.linalg.matrix_rank(B)}/{B.shape[0]}")

# Look at specific equations that might be problematic
print(f"\nChecking specific rows of A and B:")
for i in range(min(5, A.shape[0])):
    a_norm = np.linalg.norm(A[i, :])
    b_norm = np.linalg.norm(B[i, :])
    var_name = linearizer.endo_vars[i] if i < len(linearizer.endo_vars) else f"eq_{i}"
    print(f"  Eq {i:2d} ({var_name:8s}): ||A[{i}]|| = {a_norm:.6e}, ||B[{i}]|| = {b_norm:.6e}")

# Check the system A*x_{t+1} + B*x_t = 0 for forward-looking variables
print(f"\nForward-looking variable indices:")
forward_indices = []
for var in linearizer.forward_vars:
    if var in linearizer.endo_vars:
        idx = linearizer.endo_vars.index(var)
        forward_indices.append(idx)
        print(f"  {var}: index {idx}")

# Check if these variables have non-zero coefficients in A
print(f"\nForward-looking variables in A matrix:")
for idx in forward_indices:
    if idx < A.shape[1]:
        a_col_norm = np.linalg.norm(A[:, idx])
        print(f"  {linearizer.endo_vars[idx]}: ||A[:, {idx}]|| = {a_col_norm:.6e}")

# Check shock transmission
print(f"\nShock transmission (C matrix):")
shock_names = ['a', 'g', 'eps_r', 'tau_c_shock', 'tau_l_shock', 'tau_f_shock']
for j, shock in enumerate(shock_names):
    c_col_norm = np.linalg.norm(C[:, j])
    print(f"  {shock}: ||C[:, {j}]|| = {c_col_norm:.6e}")

# Look for specific Y and C equations
try:
    y_idx = linearizer.endo_vars.index('Y')
    c_idx = linearizer.endo_vars.index('C')
    
    print(f"\nY equation (row {y_idx}):")
    print(f"  A[{y_idx}, :] non-zero: {np.count_nonzero(A[y_idx, :])}")
    print(f"  B[{y_idx}, :] non-zero: {np.count_nonzero(B[y_idx, :])}")
    print(f"  Max |A[{y_idx}, :]|: {np.max(np.abs(A[y_idx, :])):.6e}")
    print(f"  Max |B[{y_idx}, :]|: {np.max(np.abs(B[y_idx, :])):.6e}")
    
    print(f"\nC equation (row {c_idx}):")
    print(f"  A[{c_idx}, :] non-zero: {np.count_nonzero(A[c_idx, :])}")
    print(f"  B[{c_idx}, :] non-zero: {np.count_nonzero(B[c_idx, :])}")
    print(f"  Max |A[{c_idx}, :]|: {np.max(np.abs(A[c_idx, :])):.6e}")
    print(f"  Max |B[{c_idx}, :]|: {np.max(np.abs(B[c_idx, :])):.6e}")
    
except ValueError as e:
    print(f"Could not find Y or C indices: {e}")

print(f"\nDiagnosis completed.")