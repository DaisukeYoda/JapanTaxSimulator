"""
Detailed debugging of linearization implementation to identify A matrix rank issues
"""

import sys
import os

# Add project root to path
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.dsge_model import DSGEModel, ModelParameters
from src.linearization_improved import ImprovedLinearizedDSGE
import sympy
import numpy as np

def debug_linearization_step_by_step():
    # Load model
    config_path = os.path.join(os.path.dirname(__file__), '../..', 'config', 'parameters.json')
    params = ModelParameters.from_json(config_path)
    model = DSGEModel(params)
    ss = model.compute_steady_state()
    
    linearizer = ImprovedLinearizedDSGE(model, ss)
    equations = linearizer.equations
    
    print("=== STEP-BY-STEP LINEARIZATION DEBUG ===")
    print(f"Number of equations: {len(equations)}")
    print(f"Number of endogenous variables: {len(linearizer.endo_vars)}")
    print(f"Endogenous variables: {linearizer.endo_vars}")
    
    # Get steady state substitutions
    ss_dict = ss.to_dict()
    
    # Test the linearization process step by step
    n_eq = len(equations)
    A = np.zeros((n_eq, len(linearizer.endo_vars)))
    B = np.zeros((n_eq, len(linearizer.endo_vars)))
    C = np.zeros((n_eq, len(linearizer.exo_vars)))
    
    print(f"\n=== PROCESSING EACH EQUATION ===")
    
    for eq_idx, equation in enumerate(equations):
        print(f"\n--- Equation {eq_idx + 1} ---")
        expr = equation.lhs - equation.rhs
        equation_symbols = expr.free_symbols
        
        print(f"Expression: {expr}")
        print(f"Symbols in equation: {[str(s) for s in equation_symbols]}")
        
        # Create substitution dictionary
        substitutions = {}
        for symbol in equation_symbols:
            symbol_str = str(symbol)
            
            if symbol_str.startswith('eps_'):
                substitutions[symbol] = 0
            else:
                # Map to steady state
                if symbol_str.endswith('_tp1'):
                    base_var = symbol_str[:-4]
                elif symbol_str.endswith('_tm1'):
                    base_var = symbol_str[:-4]
                elif symbol_str.endswith('_t'):
                    base_var = symbol_str[:-2]
                else:
                    base_var = symbol_str
                
                if base_var in ss_dict:
                    substitutions[symbol] = ss_dict[base_var]
                else:
                    substitutions[symbol] = 1.0
        
        # Check forward-looking derivatives (A matrix)
        forward_derivatives = []
        for var_idx, var in enumerate(linearizer.endo_vars):
            forward_symbol = None
            for symbol in equation_symbols:
                symbol_str = str(symbol)
                if symbol_str == f'{var}_tp1' or (symbol_str.endswith('_tp1') and symbol_str[:-4] == var):
                    forward_symbol = symbol
                    break
            
            if forward_symbol:
                try:
                    deriv = sympy.diff(expr, forward_symbol)
                    if not deriv.is_zero:
                        deriv_val = float(deriv.subs(substitutions))
                        A[eq_idx, var_idx] = deriv_val
                        forward_derivatives.append((var, forward_symbol, deriv_val))
                except Exception as e:
                    print(f"    ERROR differentiating w.r.t. {forward_symbol}: {e}")
        
        if forward_derivatives:
            print(f"Forward derivatives (A matrix entries):")
            for var, symbol, val in forward_derivatives:
                print(f"    d/d{symbol}: {val}")
        else:
            print(f"No forward derivatives found")
        
        # Check current period derivatives (B matrix)
        current_derivatives = []
        for var_idx, var in enumerate(linearizer.endo_vars):
            current_symbol = None
            for symbol in equation_symbols:
                symbol_str = str(symbol)
                if (symbol_str == f'{var}_t' or 
                    (symbol_str.endswith('_t') and symbol_str[:-2] == var) or
                    symbol_str == var):
                    current_symbol = symbol
                    break
            
            if current_symbol:
                try:
                    deriv = sympy.diff(expr, current_symbol)
                    if not deriv.is_zero:
                        deriv_val = float(deriv.subs(substitutions))
                        B[eq_idx, var_idx] = deriv_val
                        current_derivatives.append((var, current_symbol, deriv_val))
                except Exception as e:
                    print(f"    ERROR differentiating w.r.t. {current_symbol}: {e}")
        
        if current_derivatives:
            print(f"Current derivatives (B matrix entries):")
            for var, symbol, val in current_derivatives[:5]:  # Show first 5
                print(f"    d/d{symbol}: {val}")
            if len(current_derivatives) > 5:
                print(f"    ... and {len(current_derivatives) - 5} more")
        else:
            print(f"No current derivatives found")
        
        # Check for exogenous derivatives (C matrix)
        shock_derivatives = []
        for shock_idx, shock in enumerate(linearizer.exo_vars):
            for symbol in equation_symbols:
                if str(symbol) == shock:
                    try:
                        deriv = sympy.diff(expr, symbol)
                        if not deriv.is_zero:
                            deriv_val = float(deriv.subs(substitutions))
                            C[eq_idx, shock_idx] = deriv_val
                            shock_derivatives.append((shock, deriv_val))
                    except Exception as e:
                        print(f"    ERROR differentiating w.r.t. {shock}: {e}")
        
        if shock_derivatives:
            print(f"Shock derivatives (C matrix entries):")
            for shock, val in shock_derivatives:
                print(f"    d/d{shock}: {val}")
    
    print(f"\n=== FINAL MATRIX ANALYSIS ===")
    print(f"A matrix shape: {A.shape}")
    print(f"A matrix rank: {np.linalg.matrix_rank(A)}/{A.shape[0]}")
    print(f"B matrix rank: {np.linalg.matrix_rank(B)}/{B.shape[0]}")
    
    # Identify zero rows in A
    zero_rows_A = []
    for i in range(A.shape[0]):
        if np.allclose(A[i, :], 0, atol=1e-12):
            zero_rows_A.append(i)
    
    print(f"Zero rows in A: {len(zero_rows_A)}/{A.shape[0]}")
    if zero_rows_A:
        print(f"Zero row indices: {zero_rows_A}")
        for row in zero_rows_A[:5]:  # Show first 5
            print(f"  Equation {row + 1}: {equations[row]}")
    
    # Check if the issue is with overdetermined system removal
    if A.shape[0] > A.shape[1]:
        print(f"\nSystem is overdetermined ({A.shape[0]} equations, {A.shape[1]} variables)")
        print("This triggers equation removal logic...")
        
        # Show which equations would be kept
        equation_norms = np.linalg.norm(np.column_stack([A, B]), axis=1)
        critical_equations = [8, 9]  # From the implementation
        
        keep_indices = []
        for eq_idx in critical_equations:
            if eq_idx < len(equation_norms):
                keep_indices.append(eq_idx)
        
        remaining_indices = [i for i in range(len(equation_norms)) if i not in keep_indices]
        remaining_norms = [equation_norms[i] for i in remaining_indices]
        sorted_remaining = sorted(zip(remaining_norms, remaining_indices), reverse=True)
        
        needed_equations = A.shape[1] - len(keep_indices)
        for _, eq_idx in sorted_remaining[:needed_equations]:
            keep_indices.append(eq_idx)
        
        keep_indices = np.sort(keep_indices)
        print(f"Would keep equations: {keep_indices + 1}")
        
        # Check if forward-looking equations are being removed
        forward_eq_indices = [2, 4, 5, 7, 13, 20]  # 0-indexed
        removed_forward_eqs = [i for i in forward_eq_indices if i not in keep_indices]
        if removed_forward_eqs:
            print(f"WARNING: Forward-looking equations being removed: {[i+1 for i in removed_forward_eqs]}")

if __name__ == "__main__":
    debug_linearization_step_by_step()