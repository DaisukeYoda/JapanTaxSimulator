"""
Analyze forward-looking terms in model equations to debug A matrix rank deficiency
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

def analyze_forward_terms():
    # Load model
    config_path = os.path.join(os.path.dirname(__file__), '../..', 'config', 'parameters.json')
    params = ModelParameters.from_json(config_path)
    model = DSGEModel(params)
    ss = model.compute_steady_state()
    
    print("=== ANALYZING FORWARD-LOOKING TERMS ===")
    
    # Get model equations
    equations = model.get_model_equations()
    print(f"Total equations: {len(equations)}")
    
    # Analyze each equation for forward-looking terms
    forward_equations = []
    
    for eq_idx, equation in enumerate(equations):
        expr = equation.lhs - equation.rhs
        symbols_in_eq = expr.free_symbols
        
        # Find forward-looking variables (ending with _tp1)
        forward_vars_in_eq = []
        for symbol in symbols_in_eq:
            symbol_str = str(symbol)
            if symbol_str.endswith('_tp1'):
                forward_vars_in_eq.append(symbol_str)
        
        if forward_vars_in_eq:
            forward_equations.append({
                'eq_index': eq_idx,
                'equation': equation,
                'forward_vars': forward_vars_in_eq
            })
            print(f"\nEquation {eq_idx + 1}:")
            print(f"  Forward variables: {forward_vars_in_eq}")
            print(f"  Expression: {expr}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Equations with forward-looking terms: {len(forward_equations)}/{len(equations)}")
    
    # Test symbolic differentiation
    print(f"\n=== TESTING SYMBOLIC DIFFERENTIATION ===")
    
    linearizer = ImprovedLinearizedDSGE(model, ss)
    print(f"Endogenous variables: {linearizer.endo_vars}")
    print(f"Variable count: {len(linearizer.endo_vars)}")
    
    # Test differentiation on a specific equation with forward terms
    if forward_equations:
        test_eq = forward_equations[0]  # Take first equation with forward terms
        eq_idx = test_eq['eq_index']
        equation = test_eq['equation']
        expr = equation.lhs - equation.rhs
        
        print(f"\nTesting equation {eq_idx + 1}:")
        print(f"Expression: {expr}")
        
        # Get steady state substitutions
        ss_dict = ss.to_dict()
        substitutions = {}
        
        symbols_in_expr = expr.free_symbols
        for symbol in symbols_in_expr:
            symbol_str = str(symbol)
            
            if symbol_str.startswith('eps_'):
                substitutions[symbol] = 0  # Shocks are zero at steady state
            else:
                # Try to map to steady state
                if symbol_str.endswith('_tp1'):
                    base_var = symbol_str[:-4]
                elif symbol_str.endswith('_tm1'):
                    base_var = symbol_str[:-4]
                elif symbol_str.endswith('_t'):
                    base_var = symbol_str[:-2]
                else:
                    base_var = symbol_str
                
                # Map to steady state name
                if base_var in ss_dict:
                    substitutions[symbol] = ss_dict[base_var]
                else:
                    print(f"  Warning: No steady state value for {base_var}, using default=1.0")
                    substitutions[symbol] = 1.0  # Safe default for linearization analysis
        
        print(f"Substitutions: {len(substitutions)} symbols")
        
        # Test differentiation with respect to forward variables
        print(f"\nTesting forward-looking derivatives:")
        for forward_var_str in test_eq['forward_vars']:
            # Find the symbol
            forward_symbol = None
            for symbol in symbols_in_expr:
                if str(symbol) == forward_var_str:
                    forward_symbol = symbol
                    break
            
            if forward_symbol:
                try:
                    deriv = sympy.diff(expr, forward_symbol)
                    print(f"  d/d{forward_var_str}: {deriv}")
                    
                    if not deriv.is_zero:
                        deriv_val = float(deriv.subs(substitutions))
                        print(f"    Numerical value: {deriv_val}")
                    else:
                        print(f"    -> ZERO derivative!")
                        
                except Exception as e:
                    print(f"    ERROR: {e}")
        
        # Test differentiation with respect to current variables
        print(f"\nTesting current period derivatives:")
        for var in linearizer.endo_vars[:5]:  # Test first 5 variables
            # Find current period symbol
            current_symbols = []
            for symbol in symbols_in_expr:
                symbol_str = str(symbol)
                if (symbol_str == f'{var}_t' or 
                    (symbol_str.endswith('_t') and symbol_str[:-2] == var) or
                    symbol_str == var):
                    current_symbols.append(symbol)
            
            for current_symbol in current_symbols:
                try:
                    deriv = sympy.diff(expr, current_symbol)
                    if not deriv.is_zero:
                        deriv_val = float(deriv.subs(substitutions))
                        print(f"  d/d{current_symbol}: {deriv_val}")
                except Exception as e:
                    print(f"  d/d{current_symbol}: ERROR - {e}")

    # Analyze variable classification
    print(f"\n=== VARIABLE CLASSIFICATION ===")
    variable_info = linearizer.variable_info
    print(f"Endogenous: {variable_info['endogenous']}")
    print(f"Exogenous: {variable_info['exogenous']}")
    print(f"Predetermined (state): {variable_info['predetermined']}")
    print(f"Jump (control): {variable_info['jump']}")
    print(f"Forward-looking: {variable_info['forward_looking']}")
    print(f"Lagged: {variable_info['lagged']}")

if __name__ == "__main__":
    analyze_forward_terms()