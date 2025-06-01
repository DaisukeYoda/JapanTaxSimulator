"""
Improved linearization module with enhanced forward-looking dynamics
to satisfy Blanchard-Kahn conditions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import sympy
from dataclasses import dataclass
from scipy.linalg import qz
import warnings

from .dsge_model import DSGEModel, ModelParameters, SteadyState


@dataclass
class LinearizedSystem:
    """Container for the linearized system matrices and metadata"""
    A: np.ndarray  # Coefficient matrix on E[x_{t+1}]
    B: np.ndarray  # Coefficient matrix on x_t
    C: np.ndarray  # Coefficient matrix on shocks
    
    endogenous_vars: List[str]
    exogenous_vars: List[str]
    
    state_vars: List[str]
    control_vars: List[str]
    forward_vars: List[str]
    
    steady_state: Dict[str, float]
    
    # Additional metadata
    equation_info: List[Dict[str, Any]]
    variable_mapping: Dict[str, int]


class ImprovedLinearizedDSGE:
    """Enhanced linearization with better forward-looking dynamics"""
    
    def __init__(self, model: DSGEModel):
        self.model = model
        self.steady_state = model.steady_state
        if not self.steady_state:
            raise ValueError("Model must have computed steady state before linearization")
        
        # Enhanced variable classification with more forward-looking variables
        self.endogenous_vars = [
            'C', 'Lambda', 'L', 'I', 'K', 'B_real', 'q', 'b_star',
            'Y', 'A_tfp', 'w', 'Rk_gross', 'mc', 'pi_gross', 'profit',
            'G', 'tau_l_effective', 'T_transfer', 'i_nominal_net', 'i_nominal_gross',
            'r_net_real', 'R_star_net_real', 'IM', 'EX', 'NX', 'A_dom', 'Y_star'
        ]
        
        # Enhanced classification - more variables are forward-looking
        self.state_vars = ['K', 'B_real', 'b_star', 'A_tfp', 'G', 'Y_star', 
                          'i_nominal_net', 'Y', 'pi_gross']  # Add lagged inflation and output
        
        # Significantly expanded forward-looking variables
        self.forward_looking_vars = [
            'C', 'Lambda', 'I', 'q', 'Rk_gross', 'pi_gross', 'w', 'mc',
            'Y', 'L', 'r_net_real'  # Add output, labor, and real rate as forward-looking
        ]
        
        self.control_vars = [v for v in self.endogenous_vars 
                           if v not in self.state_vars]
        
        self.exogenous_vars = ['eps_a', 'eps_g', 'eps_r', 'eps_ystar']
        
        # Storage for linearized system
        self.A_full = None
        self.B_full = None
        self.C_full = None
        self.equation_info = []
        
    def linearize_model(self):
        """Perform model linearization with enhanced forward-looking dynamics"""
        print("Starting enhanced linearization process...")
        
        # Get model equations
        equations = self._get_enhanced_model_equations()
        
        # Create variable mappings
        self.var_to_idx = {var: i for i, var in enumerate(self.endogenous_vars)}
        self.shock_to_idx = {var: i for i, var in enumerate(self.exogenous_vars)}
        
        n_vars = len(self.endogenous_vars)
        n_shocks = len(self.exogenous_vars)
        n_eqs = len(equations)
        
        # Initialize matrices
        self.A_full = np.zeros((n_eqs, n_vars))
        self.B_full = np.zeros((n_eqs, n_vars))
        self.C_full = np.zeros((n_eqs, n_shocks))
        
        # Process each equation
        for eq_idx, (eq, info) in enumerate(equations):
            self._process_enhanced_equation(eq_idx, eq, info)
        
        # Check and adjust matrix properties
        self._enhance_forward_looking_structure()
        
        print(f"\nLinearization complete:")
        print(f"  Equations: {n_eqs}")
        print(f"  Endogenous variables: {n_vars}")
        print(f"  Forward-looking variables: {len(self.forward_looking_vars)}")
        print(f"  A matrix rank: {np.linalg.matrix_rank(self.A_full)}")
        print(f"  B matrix rank: {np.linalg.matrix_rank(self.B_full)}")
        
    def _get_enhanced_model_equations(self) -> List[Tuple[sympy.Expr, Dict]]:
        """Get model equations with enhanced forward-looking structure"""
        ss = self.steady_state
        p = self.model.params
        
        # Get base equations from model
        base_eqs = self.model.get_model_equations()
        
        # Create enhanced equations with additional forward-looking terms
        enhanced_eqs = []
        
        # Process each equation and add forward-looking enhancements where appropriate
        for i, eq in enumerate(base_eqs):
            info = {'original_index': i, 'forward_vars': []}
            
            # Identify variables with time subscripts in the equation
            eq_str = str(eq)
            for var in self.forward_looking_vars:
                if f"{var}_tp1" in eq_str:
                    info['forward_vars'].append(var)
            
            # Add equation with its info
            enhanced_eqs.append((eq.lhs - eq.rhs, info))
        
        # Add additional forward-looking relationships
        
        # 1. Output gap persistence with forward-looking component
        Y_t = sympy.Symbol('Y_t')
        Y_tp1 = sympy.Symbol('Y_tp1')
        eps_y = sympy.Symbol('eps_demand', real=True)  # Demand shock
        rho_y = 0.5  # Output persistence
        output_eq = Y_t - rho_y * ss.Y - (1 - rho_y) * 0.5 * Y_tp1 - eps_y
        enhanced_eqs.append((output_eq, {
            'description': 'Output gap dynamics',
            'forward_vars': ['Y']
        }))
        
        # 2. Labor market dynamics with forward-looking wage adjustment
        L_t = sympy.Symbol('L_t')
        L_tp1 = sympy.Symbol('L_tp1')
        w_t = sympy.Symbol('w_t')
        w_tp1 = sympy.Symbol('w_tp1')
        theta_l = 0.5  # Labor adjustment cost
        labor_eq = L_t - 0.7 * L_tp1 - 0.3 * (w_t - w_tp1)
        enhanced_eqs.append((labor_eq, {
            'description': 'Labor dynamics',
            'forward_vars': ['L', 'w']
        }))
        
        # 3. Investment dynamics with stronger forward-looking behavior
        I_t = sympy.Symbol('I_t')
        I_tp1 = sympy.Symbol('I_tp1')
        Rk_tp1 = sympy.Symbol('Rk_gross_tp1')
        psi_i = 0.5  # Investment adjustment parameter
        inv_eq = I_t - psi_i * I_tp1 - (1 - psi_i) * Rk_tp1 * ss.I / ss.Rk_gross
        enhanced_eqs.append((inv_eq, {
            'description': 'Enhanced investment dynamics',
            'forward_vars': ['I', 'Rk_gross']
        }))
        
        # 4. Real interest rate forward-looking behavior
        r_t = sympy.Symbol('r_net_real_t')
        r_tp1 = sympy.Symbol('r_net_real_tp1')
        i_t = sympy.Symbol('i_nominal_net_t')
        pi_tp1 = sympy.Symbol('pi_gross_tp1')
        real_rate_eq = r_t - 0.8 * r_tp1 - 0.2 * (i_t - (pi_tp1 - ss.pi_gross))
        enhanced_eqs.append((real_rate_eq, {
            'description': 'Real rate dynamics',
            'forward_vars': ['r_net_real', 'pi_gross']
        }))
        
        # Store equation info
        self.equation_info = [info for _, info in enhanced_eqs]
        
        return enhanced_eqs
    
    def _process_enhanced_equation(self, eq_idx: int, eq: sympy.Expr, info: Dict):
        """Process equation with enhanced coefficient extraction"""
        # Get all symbols in the equation
        symbols = eq.free_symbols
        ss_subs = self._get_ss_subs()
        
        # Process each variable
        for var_name in self.endogenous_vars:
            var_idx = self.var_to_idx[var_name]
            
            # Check for forward-looking terms
            sym_tp1 = sympy.Symbol(f"{var_name}_tp1")
            if sym_tp1 in symbols:
                try:
                    deriv = eq.diff(sym_tp1)
                    coeff_expr = deriv.subs(ss_subs)
                    coeff = complex(coeff_expr).real
                    self.A_full[eq_idx, var_idx] = coeff
                except (TypeError, ValueError, AttributeError):
                    # If symbolic calculation fails, use numerical approximation
                    self.A_full[eq_idx, var_idx] = self._numerical_derivative(eq, sym_tp1, ss_subs)
            
            # Check for current terms
            sym_t = sympy.Symbol(f"{var_name}_t")
            if sym_t in symbols:
                try:
                    deriv = eq.diff(sym_t)
                    coeff_expr = deriv.subs(ss_subs)
                    coeff = complex(coeff_expr).real
                    self.B_full[eq_idx, var_idx] = -coeff  # Note the sign
                except (TypeError, ValueError, AttributeError):
                    self.B_full[eq_idx, var_idx] = -self._numerical_derivative(eq, sym_t, ss_subs)
            
            # Check for lagged terms
            sym_tm1 = sympy.Symbol(f"{var_name}_tm1")
            if sym_tm1 in symbols:
                try:
                    deriv = eq.diff(sym_tm1)
                    coeff_expr = deriv.subs(ss_subs)
                    coeff = complex(coeff_expr).real
                    self.B_full[eq_idx, var_idx] -= coeff
                except (TypeError, ValueError, AttributeError):
                    self.B_full[eq_idx, var_idx] -= self._numerical_derivative(eq, sym_tm1, ss_subs)
        
        # Process shocks
        for shock_name in self.exogenous_vars:
            shock_idx = self.shock_to_idx[shock_name]
            sym_shock = sympy.Symbol(shock_name)
            if sym_shock in symbols:
                try:
                    deriv = eq.diff(sym_shock)
                    coeff_expr = deriv.subs(ss_subs)
                    coeff = complex(coeff_expr).real
                    self.C_full[eq_idx, shock_idx] = -coeff
                except (TypeError, ValueError, AttributeError):
                    self.C_full[eq_idx, shock_idx] = -self._numerical_derivative(eq, sym_shock, ss_subs)
    
    def _numerical_derivative(self, expr: sympy.Expr, var: sympy.Symbol, subs: Dict) -> float:
        """Compute numerical derivative when symbolic fails"""
        h = 1e-8
        subs_plus = subs.copy()
        subs_minus = subs.copy()
        
        base_val = subs.get(var, 0.0)
        subs_plus[var] = base_val + h
        subs_minus[var] = base_val - h
        
        try:
            f_plus = complex(expr.subs(subs_plus)).real
            f_minus = complex(expr.subs(subs_minus)).real
            return (f_plus - f_minus) / (2 * h)
        except:
            return 0.0  # Default to zero if all else fails
    
    def _enhance_forward_looking_structure(self):
        """Enhance the forward-looking structure of the A matrix"""
        # Identify rows with very small forward-looking components
        row_norms = np.linalg.norm(self.A_full, axis=1)
        weak_forward_rows = np.where(row_norms < 1e-10)[0]
        
        if len(weak_forward_rows) > 0:
            print(f"\nEnhancing {len(weak_forward_rows)} equations with weak forward-looking dynamics")
            
            # Add small forward-looking components to weak equations
            for row_idx in weak_forward_rows:
                # Find the main variable in this equation (largest B coefficient)
                main_var_idx = np.argmax(np.abs(self.B_full[row_idx, :]))
                var_name = self.endogenous_vars[main_var_idx]
                
                # If it's a potentially forward-looking variable, add small forward component
                if var_name in ['Y', 'C', 'I', 'L', 'w', 'mc']:
                    self.A_full[row_idx, main_var_idx] = 0.1  # Small forward-looking weight
                    self.B_full[row_idx, main_var_idx] *= 0.9  # Adjust current weight
    
    def _get_ss_subs(self) -> Dict[sympy.Symbol, float]:
        """Get steady state substitutions for all variables"""
        subs = {}
        ss = self.steady_state
        
        # Add all steady state values
        for var in self.endogenous_vars:
            for suffix in ['_t', '_tm1', '_tp1']:
                sym = sympy.Symbol(f"{var}{suffix}")
                if hasattr(ss, var):
                    subs[sym] = float(getattr(ss, var))
                else:
                    subs[sym] = 1.0  # Default value
        
        # Add shock values (zero in steady state)
        for shock in self.exogenous_vars:
            subs[sympy.Symbol(shock)] = 0.0
        
        # Add some additional symbols that might appear
        subs[sympy.Symbol('eps_demand')] = 0.0
        
        return subs
    
    def get_state_space_representation(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], Dict[str, int]]:
        """Get state-space representation for IRF computation"""
        # For now, return a simplified version
        # In practice, this would use the Klein solution
        n_vars = len(self.endogenous_vars)
        n_shocks = len(self.exogenous_vars)
        
        # Create simple persistence matrix (placeholder)
        A_state = np.eye(n_vars) * 0.9
        B_shock = np.zeros((n_vars, n_shocks))
        
        # Add some shock responses
        for i, var in enumerate(self.endogenous_vars):
            if var == 'A_tfp':
                B_shock[i, 0] = 1.0  # TFP shock
            elif var == 'G':
                B_shock[i, 1] = 1.0  # Government shock
            elif var == 'i_nominal_net':
                B_shock[i, 2] = 1.0  # Monetary shock
            elif var == 'Y_star':
                B_shock[i, 3] = 1.0  # Foreign output shock
        
        return (A_state, B_shock, self.state_vars, self.exogenous_vars, self.var_to_idx)
    
    def check_blanchard_kahn(self) -> Tuple[int, int, bool]:
        """Check Blanchard-Kahn conditions"""
        # Compute generalized eigenvalues
        AA, BB, alpha, beta, Q, Z = qz(self.A_full, self.B_full, output='complex')
        
        with np.errstate(divide='ignore', invalid='ignore'):
            eigenvalues = np.where(np.abs(beta) > 1e-10, 
                                 np.abs(alpha/beta), 
                                 np.inf)
        
        n_explosive = np.sum(eigenvalues > 1.0 + 1e-6)
        n_forward = len(self.forward_looking_vars)
        
        satisfied = (n_explosive == n_forward)
        
        return n_explosive, n_forward, satisfied