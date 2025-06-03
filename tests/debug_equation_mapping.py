#!/usr/bin/env python3
"""
Debug equation mapping to understand why eps_a is mapped to wrong equation
"""

import numpy as np
import sympy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dsge_model import DSGEModel, ModelParameters
from src.linearization_improved import ImprovedLinearizedDSGE

def debug_equation_mapping():
    print("=== Debugging Equation Mapping ===\n")
    
    # Load model
    params = ModelParameters.from_json('config/parameters.json')
    model = DSGEModel(params)
    ss = model.compute_steady_state()
    
    # Get equations
    equations = model.get_model_equations()
    print(f"Total equations: {len(equations)}")
    
    # Find TFP-related equations
    print("\n1. TFP-Related Equations:")
    for i, eq in enumerate(equations):
        eq_str = str(eq)
        if 'A_tfp' in eq_str:
            print(f"\nEquation {i}: {eq_str[:100]}...")
            # Check if eps_a is in this equation
            if 'eps_a' in eq_str:
                print(f"   ✓ Contains eps_a shock")
            else:
                print(f"   ✗ No eps_a shock")
                
    # Look specifically at equation 10 (TFP evolution)
    print("\n2. Equation 10 (TFP evolution) Analysis:")
    eq10 = equations[10]
    print(f"   Original: {eq10}")
    
    # Check symbols in equation 10
    symbols = eq10.free_symbols
    print(f"\n   Symbols in equation: {[str(s) for s in symbols]}")
    
    # Create linearizer and check how equation 10 is processed
    linearizer = ImprovedLinearizedDSGE(model, ss)
    
    # Build matrices with debug output
    print("\n3. Building System Matrices...")
    
    # Manually process equation 10 to see what happens
    eq10_expr = eq10.lhs - eq10.rhs
    print(f"\n   Equation 10 expression: {eq10_expr}")
    
    # Get steady state substitutions
    ss_dict = ss.to_dict()
    substitutions = {}
    for symbol in eq10_expr.free_symbols:
        symbol_str = str(symbol)
        # Handle different variable name patterns
        if symbol_str.endswith('_t') or symbol_str.endswith('_tp1') or symbol_str.endswith('_tm1'):
            # Extract base name
            if symbol_str.endswith('_tp1'):
                base_name = symbol_str[:-4]
            elif symbol_str.endswith('_tm1'):
                base_name = symbol_str[:-4]
            elif symbol_str.endswith('_t'):
                base_name = symbol_str[:-2]
            else:
                base_name = symbol_str
                
            # Look for steady state value
            if base_name in ss_dict:
                substitutions[symbol] = ss_dict[base_name]
    
    print(f"\n   Substitutions: {substitutions}")
    
    # Try to differentiate with respect to eps_a
    eps_a_symbol = None
    for sym in eq10_expr.free_symbols:
        if str(sym) == 'eps_a':
            eps_a_symbol = sym
            break
    
    if eps_a_symbol:
        print(f"\n   Found eps_a symbol: {eps_a_symbol}")
        deriv = sympy.diff(eq10_expr, eps_a_symbol)
        print(f"   Derivative w.r.t. eps_a: {deriv}")
        
        if not deriv.is_zero:
            try:
                deriv_val = float(deriv.subs(substitutions))
                print(f"   Numerical value: {deriv_val}")
            except Exception as e:
                print(f"   Failed to evaluate: {e}")
    else:
        print(f"\n   ✗ eps_a symbol not found in equation!")
        
    # Check how the full system processes equations
    print("\n4. Full System Matrix Building:")
    linear_system = linearizer.build_system_matrices()
    
    # Check which equations are kept/removed
    print(f"\n   Original equations: 29")
    print(f"   Final equations: {linear_system.A.shape[0]}")
    
    # Try to map which original equations correspond to final rows
    print("\n5. Equation Removal Analysis:")
    
    # The code shows equations 2-17, 19-29 are kept (0-indexed becomes 1-16, 18-28)
    kept_equations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
    removed_equations = [0, 17]
    
    print(f"   Kept equations (0-indexed): {kept_equations}")
    print(f"   Removed equations (0-indexed): {removed_equations}")
    
    # Check if equation 10 (TFP evolution) is in kept list
    if 10 in kept_equations:
        new_index = kept_equations.index(10)
        print(f"\n   Equation 10 (TFP evolution) maps to row {new_index} in final system")
    else:
        print(f"\n   ✗ ERROR: Equation 10 (TFP evolution) was removed!")
        
    print("\n=== Debug Complete ===")

if __name__ == "__main__":
    debug_equation_mapping()