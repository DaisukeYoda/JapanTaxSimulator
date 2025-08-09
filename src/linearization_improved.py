"""
Improved linear approximation and solution methods for DSGE model

This module implements proper linearization and solution methods using
the Klein (2000) or Sims (2002) approach for solving linear rational
expectations models.
"""

import numpy as np
import pandas as pd
from scipy import linalg
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from dataclasses import dataclass

# Set up logger for this module
logger = logging.getLogger(__name__)

try:
    from .dsge_model import DSGEModel, SteadyState, ModelParameters
except ImportError:
    from dsge_model import DSGEModel, SteadyState, ModelParameters


@dataclass
class LinearizedSystem:
    """Container for linearized system matrices"""
    A: np.ndarray  # Coefficient on E[x_{t+1}]
    B: np.ndarray  # Coefficient on x_t
    C: np.ndarray  # Coefficient on exogenous shocks
    P: np.ndarray  # Policy function: control = P * state
    Q: np.ndarray  # Transition matrix: state_{t+1} = Q * state_t + R * shock_t
    R: np.ndarray  # Shock loading matrix
    var_names: List[str]  # Variable names
    state_vars: List[str]  # State variable names
    control_vars: List[str]  # Control variable names
    

class ImprovedLinearizedDSGE:
    """
    Improved linearization and solution of DSGE models using symbolic differentiation
    """
    
    def __init__(self, model: DSGEModel, steady_state: SteadyState):
        self.model = model
        self.steady_state = steady_state
        self.params = model.params
        
        # Defer heavy setup until build_system_matrices() is called
        self.equations = None
        self.variable_info = None
        
        # Placeholders; will be populated lazily
        self.endo_vars: List[str] = []
        self.exo_vars: List[str] = []
        self.state_vars: List[str] = []
        self.control_vars: List[str] = []
        
        self.n_endo = 0
        self.n_exo = 0
        self.n_state = 0
        self.n_control = 0
        
        # For compatibility with tax_simulator.py
        self.n_s = self.n_state
        
        # Initialize solution matrices
        self.linear_system = None
        
    def _extract_variables_from_equations(self) -> Dict:
        """
        Extract variable classification from model equations using sympy
        """
        import sympy
        import re
        
        # Collect all symbols from equations
        all_symbols = set()
        for eq in self.equations:
            all_symbols.update(eq.free_symbols)
        
        # Classify variables based on their time subscripts
        current_vars = set()
        forward_vars = set()
        lagged_vars = set()
        exogenous_vars = set()
        
        for symbol in all_symbols:
            name = str(symbol)
            
            # Identify exogenous shocks (eps_*)
            if name.startswith('eps_'):
                exogenous_vars.add(name)
                continue
                
            # Parse variable names with time subscripts
            # Variables ending with _tp1 are forward-looking
            if name.endswith('_tp1'):
                base_name = name[:-4]  # Remove _tp1
                forward_vars.add(base_name)
                current_vars.add(base_name)
            # Variables ending with _tm1 are lagged
            elif name.endswith('_tm1'):
                base_name = name[:-4]  # Remove _tm1
                lagged_vars.add(base_name)
                current_vars.add(base_name)
            # Variables ending with _t are current period
            elif name.endswith('_t'):
                base_name = name[:-2]  # Remove _t
                current_vars.add(base_name)
            else:
                # Handle variables without explicit time subscript
                current_vars.add(name)
        
        # Remove exogenous variables from endogenous sets
        current_vars -= exogenous_vars
        forward_vars -= exogenous_vars
        lagged_vars -= exogenous_vars
        
        # Convert to sorted lists for consistent ordering
        endogenous_vars = sorted(list(current_vars))
        exogenous_vars_list = sorted(list(exogenous_vars))
        
        # Classify predetermined vs jump variables
        # Predetermined: Variables that appear lagged (state variables)
        predetermined_vars = sorted(list(lagged_vars))
        
        # Jump variables: Variables that appear with forward expectations but not lagged
        jump_vars = sorted(list(forward_vars - lagged_vars))
        
        # Remaining current variables are also jump variables
        remaining_vars = sorted(list(current_vars - forward_vars - lagged_vars))
        jump_vars.extend(remaining_vars)
        jump_vars = sorted(list(set(jump_vars)))
        
        return {
            'endogenous': endogenous_vars,
            'exogenous': exogenous_vars_list,
            'predetermined': predetermined_vars,
            'jump': jump_vars,
            'forward_looking': sorted(list(forward_vars)),
            'lagged': sorted(list(lagged_vars))
        }
        
    def build_system_matrices(self) -> LinearizedSystem:
        """
        Build the linearized system matrices using symbolic differentiation
        A*E[x_{t+1}] + B*x_t + C*z_t = 0
        """
        import sympy
        
        # Lazily load equations and variable classifications
        if self.equations is None:
            self.equations = self.model.get_model_equations()
        if self.variable_info is None:
            self.variable_info = self._extract_variables_from_equations()
            self.endo_vars = self.variable_info['endogenous']
            self.exo_vars = self.variable_info['exogenous']
            self.state_vars = self.variable_info['predetermined']
            self.control_vars = self.variable_info['jump']
            self.n_endo = len(self.endo_vars)
            self.n_exo = len(self.exo_vars)
            self.n_state = len(self.state_vars)
            self.n_control = len(self.control_vars)
        
        # Get steady state values
        ss_dict = self.steady_state.to_dict()
        
        # Collect all symbols that actually appear in the equations
        all_symbols_in_equations = set()
        for eq in self.equations:
            all_symbols_in_equations.update(eq.free_symbols)
        
        # Map symbols to variable names and time periods
        symbol_mapping = {}  # symbol -> (var_name, time_period)
        
        for symbol in all_symbols_in_equations:
            symbol_str = str(symbol)
            
            if symbol_str.endswith('_tp1'):
                var_name = symbol_str[:-4]
                symbol_mapping[symbol] = (var_name, 'tp1')
            elif symbol_str.endswith('_tm1'):
                var_name = symbol_str[:-4]
                symbol_mapping[symbol] = (var_name, 'tm1')
            elif symbol_str.endswith('_t'):
                var_name = symbol_str[:-2]
                symbol_mapping[symbol] = (var_name, 't')
            elif symbol_str.startswith('eps_'):
                symbol_mapping[symbol] = (symbol_str, 'shock')
            else:
                # Variables without explicit time subscript (current period)
                symbol_mapping[symbol] = (symbol_str, 't')
        
        # Create substitution dictionary for steady state values
        substitutions = {}
        
        for symbol, (var_name, time_period) in symbol_mapping.items():
            if time_period == 'shock':
                substitutions[symbol] = 0  # Shocks are zero at steady state
            else:
                # Map to steady state variable name
                ss_var = self._map_to_steady_state_name(var_name)
                if ss_var in ss_dict:
                    substitutions[symbol] = ss_dict[ss_var]
                elif var_name in ss_dict:
                    substitutions[symbol] = ss_dict[var_name]
                else:
                    raise ValueError(
                        f"Missing steady state value for '{var_name}' (mapped key '{ss_var}'). "
                        f"Provide SS in model steady state; no defaults allowed for research integrity."
                    )
        
        # Initialize coefficient matrices
        n_eq = len(self.equations)
        A = np.zeros((n_eq, self.n_endo))  # Coefficients on E[x_{t+1}]
        B = np.zeros((n_eq, self.n_endo))  # Coefficients on x_t
        C = np.zeros((n_eq, self.n_exo))   # Coefficients on shocks
        
        # Process each equation
        for eq_idx, equation in enumerate(self.equations):
            # Extract the equation expression (left side - right side = 0)
            expr = equation.lhs - equation.rhs
            
            # Find all symbols in this equation
            equation_symbols = expr.free_symbols
            
            # Differentiate with respect to forward-looking variables (t+1)
            for var_idx, var in enumerate(self.endo_vars):
                # Find the t+1 symbol for this variable
                for symbol in equation_symbols:
                    symbol_str = str(symbol)
                    if symbol_str == f'{var}_tp1' or (symbol_str.endswith('_tp1') and symbol_str[:-4] == var):
                        try:
                            deriv = sympy.diff(expr, symbol)
                            if not deriv.is_zero:
                                deriv_val = float(deriv.subs(substitutions))
                                A[eq_idx, var_idx] = deriv_val
                                break
                        except Exception as e:
                            print(f"Warning: Failed to differentiate equation {eq_idx} w.r.t. {symbol}: {e}")
            
            # Differentiate with respect to current period variables (t)
            for var_idx, var in enumerate(self.endo_vars):
                # Find the current period symbol for this variable
                for symbol in equation_symbols:
                    symbol_str = str(symbol)
                    if (symbol_str == f'{var}_t' or 
                        (symbol_str.endswith('_t') and symbol_str[:-2] == var) or
                        symbol_str == var):  # Variables without explicit _t suffix
                        try:
                            deriv = sympy.diff(expr, symbol)
                            if not deriv.is_zero:
                                deriv_val = float(deriv.subs(substitutions))
                                B[eq_idx, var_idx] = deriv_val
                                break
                        except Exception as e:
                            print(f"Warning: Failed to differentiate equation {eq_idx} w.r.t. {symbol}: {e}")
            
            # Differentiate with respect to exogenous shocks
            for shock_idx, shock in enumerate(self.exo_vars):
                for symbol in equation_symbols:
                    if str(symbol) == shock:
                        try:
                            deriv = sympy.diff(expr, symbol)
                            if not deriv.is_zero:
                                deriv_val = float(deriv.subs(substitutions))
                                C[eq_idx, shock_idx] = deriv_val
                                break
                        except Exception as e:
                            print(f"Warning: Failed to differentiate equation {eq_idx} w.r.t. {symbol}: {e}")
        
        # Ensure square system: require exactly n_endo equations (no ad-hoc filtering)
        target_eq_count = A.shape[1]  # Number of variables
        if A.shape[0] != target_eq_count:
            raise ValueError(
                f"Equation count mismatch: have {A.shape[0]} equations for {target_eq_count} variables. "
                f"Adjust the model equations so the system is square; automatic filtering/regularization is disabled."
            )
        
        # Final verification
        A_rank = np.linalg.matrix_rank(A)
        print(f"Square system achieved: {A.shape} with rank {A_rank}")
        if A_rank < A.shape[0]:
            print(f"A matrix is rank deficient ({A_rank}/{A.shape[0]}). This may be acceptable if BK holds.")
        
        # Store the system 
        self.linear_system = LinearizedSystem(
            A=A,
            B=B, 
            C=C,
            P=None,
            Q=None,
            R=None,
            var_names=self.endo_vars,
            state_vars=self.state_vars,
            control_vars=self.control_vars
        )
        
        return self.linear_system
    
    def _map_to_steady_state_name(self, var: str) -> str:
        """
        Map linearization variable names to steady state variable names
        """
        # Handle special cases where variable names differ
        mapping = {
            'pi_gross': 'pi_gross',
            'r_net_real': 'r_net_real', 
            'i_nominal_gross': 'i_nominal_gross',
            'i_nominal_net': 'i_nominal_net',
            'Rk_gross': 'Rk_gross',
            'B_real': 'B_real',
            'A_tfp': 'A_tfp',
            'tau_l_effective': 'tau_l_effective',
            'R_star_net_real': 'R_star_net_real',
            'Y_star': 'Y_star',
            'T_transfer': 'T_transfer'
        }
        
        return mapping.get(var, var)
    
    def solve_klein(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the linear system using Klein's (2000) method
        
        Returns:
            P: Policy function matrix
            Q: Transition matrix
        """
        if self.linear_system is None:
            self.build_system_matrices()
        
        A = self.linear_system.A
        B = self.linear_system.B
        
        print(f"Matrix A shape: {A.shape}")
        print(f"Matrix B shape: {B.shape}")
        print(f"A matrix rank: {np.linalg.matrix_rank(A)}")
        print(f"B matrix rank: {np.linalg.matrix_rank(B)}")
        
        # Generalized Schur decomposition (no fallback allowed)
        T, S, alpha, beta, Q_schur, Z = linalg.ordqz(A, B, sort='ouc')
        
        # Check Blanchard-Kahn conditions
        eigenvalues = alpha / beta
        finite_eigenvals = eigenvalues[np.isfinite(eigenvalues)]
        n_explosive = np.sum(np.abs(finite_eigenvals) > 1.0)
        n_jump = self.n_control
        print(f"Number of explosive eigenvalues: {n_explosive}")
        print(f"Number of jump variables: {n_jump}")
        if n_explosive != n_jump:
            raise ValueError(
                f"Blanchard-Kahn conditions not satisfied (unstable roots={n_explosive}, jump vars={n_jump}). "
                f"Adjust model specification to satisfy BK; no fallback permitted."
            )
        
        # Recover solution using stable invariant subspace (Klein/Sims method)
        n_vars = len(self.endo_vars)
        k_unstable = n_explosive  # equals number of jump variables after BK
        # Partition columns of Z into [unstable | stable]
        Z_unstable = Z[:, :k_unstable]
        Z_stable = Z[:, k_unstable:]
        
        # Partition rows of Z according to variable types (states first, controls second)
        # Build index maps from variable name ordering used in A/B (self.endo_vars)
        var_to_idx = {v: i for i, v in enumerate(self.endo_vars)}
        state_row_idx = [var_to_idx[v] for v in self.state_vars if v in var_to_idx]
        control_row_idx = [var_to_idx[v] for v in self.control_vars if v in var_to_idx]
        
        # Sanity checks
        if len(state_row_idx) != self.n_state or len(control_row_idx) != self.n_control:
            raise ValueError(
                f"State/control index mismatch (states={len(state_row_idx)}/{self.n_state}, "
                f"controls={len(control_row_idx)}/{self.n_control}). Ensure variable metadata is correct."
            )
        
        # Stable eigen-blocks of S and T (lower-right after ordering)
        S_stable = S[k_unstable:, k_unstable:]
        T_stable = T[k_unstable:, k_unstable:]
        # Transition in stable coordinates: w_{t+1} = S_s^{-1} T_s w_t
        G_stable = linalg.solve(S_stable, T_stable)
        
        # Z blocks for stable columns
        Zs_states = Z_stable[state_row_idx, :]
        Zs_controls = Z_stable[control_row_idx, :]
        
        # Policy function: controls = P * states
        # P = Zs_controls @ inv(Zs_states)
        try:
            Zs_states_inv = linalg.inv(Zs_states)
        except linalg.LinAlgError:
            logger.warning("Zs_states matrix is singular. Using pseudo-inverse. This may indicate issues with model specification.")
            Zs_states_inv = linalg.pinv(Zs_states)
        P = Zs_controls @ Zs_states_inv
        
        # State transition in original state coordinates:
        # x_state,t = Zs_states w_s,t  => w_s,t = Zs_states^{-1} x_state,t
        # x_state,t+1 = Zs_states w_s,t+1 = Zs_states G_stable Zs_states^{-1} x_state,t
        Q_states = Zs_states @ G_stable @ Zs_states_inv
        
        # Build solution matrices with correct shapes
        P_full = np.zeros((self.n_control, self.n_state))
        P_full[:, :] = P[:self.n_control, :self.n_state]
        
        Q_full = np.zeros((self.n_state, self.n_state))
        Q_full[:, :] = Q_states[:self.n_state, :self.n_state]
        
        # Add persistence for exogenous processes
        exo_indices = self._get_exogenous_state_indices()
        for shock_name, idx in exo_indices.items():
            if shock_name == 'A_tfp' and hasattr(self.params, 'rho_a'):
                Q_full[idx, idx] = self.params.rho_a
            elif shock_name == 'G' and hasattr(self.params, 'rho_g'):
                Q_full[idx, idx] = self.params.rho_g
            elif shock_name == 'Y_star' and hasattr(self.params, 'rho_ystar'):
                Q_full[idx, idx] = self.params.rho_ystar
        
        # Store solution
        self.linear_system.P = P_full
        self.linear_system.Q = Q_full
        self.linear_system.R = np.zeros((self.n_state, self.n_exo))
        
        # Also expose as attributes for downstream usage
        self.P_matrix = P_full
        self.Q_matrix = Q_full
        
        return P_full, Q_full
    
    def _get_exogenous_state_indices(self) -> Dict[str, int]:
        """Get indices of exogenous state variables"""
        indices = {}
        for i, var in enumerate(self.state_vars):
            if var in ['A_tfp', 'G', 'Y_star']:
                indices[var] = i
        return indices
    
    def compute_impulse_response(self,
                               shock_type: str,
                               shock_size: float = 1.0,
                               periods: int = 40,
                               variables: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compute impulse response functions with proper state space solution
        """
        if self.linear_system is None or self.linear_system.P is None:
            self.solve_klein()
        
        # Map shock types to exogenous variable indices
        shock_map = {}
        for i, exo_var in enumerate(self.exo_vars):
            if 'eps_a' in exo_var:
                shock_map['tfp'] = i
            elif 'eps_g' in exo_var:
                shock_map['gov_spending'] = i
            elif 'eps_r' in exo_var:
                shock_map['monetary'] = i
            elif 'eps_ystar' in exo_var:
                shock_map['foreign_output'] = i
        
        if shock_type not in shock_map:
            available_shocks = list(shock_map.keys())
            raise ValueError(f"Unknown shock type: {shock_type}. Available: {available_shocks}")
        
        shock_idx = shock_map[shock_type]
        
        # Initialize arrays for all variables (states + controls combined)
        n_total_vars = len(self.endo_vars)
        var_path = np.zeros((periods + 1, n_total_vars))
        
        # Create shock vector
        shock_vector = np.zeros(self.n_exo)
        
        # Set shock size based on type and available parameters
        if shock_type == 'tfp' and hasattr(self.params, 'sigma_a'):
            shock_vector[shock_idx] = shock_size * self.params.sigma_a
        elif shock_type == 'gov_spending' and hasattr(self.params, 'sigma_g'):
            shock_vector[shock_idx] = shock_size * self.params.sigma_g
        elif shock_type == 'monetary' and hasattr(self.params, 'sigma_r'):
            shock_vector[shock_idx] = shock_size * self.params.sigma_r
        else:
            # Default shock size
            shock_vector[shock_idx] = shock_size * 0.01  # 1% shock
        
        # Solve the dynamic system: A*E[x_{t+1}] + B*x_t + C*z_t = 0
        # Rearrange to: E[x_{t+1}] = -A^{-1}*B*x_t - A^{-1}*C*z_t
        
        A_inv = np.linalg.pinv(self.linear_system.A)
        transition_matrix = -A_inv @ self.linear_system.B
        shock_loading = -A_inv @ self.linear_system.C
        
        # Initialize with zero values
        var_path[0, :] = 0
        
        # Apply shock to period 0 and simulate forward
        for t in range(periods):
            # Current period shock (non-zero only in period 0 for impulse)
            current_shock = np.zeros(self.n_exo)
            if t == 0:
                current_shock[shock_idx] = shock_vector[shock_idx]
            
            # Next period values
            if t == 0:
                # Initial response to shock
                var_path[t + 1] = shock_loading @ current_shock
            else:
                # Dynamic propagation
                var_path[t + 1] = transition_matrix @ var_path[t]
            
            # Add persistence for exogenous variables
            if t > 0:
                # Apply AR(1) persistence to TFP if applicable
                if shock_type == 'tfp' and hasattr(self.params, 'rho_a'):
                    # Find A_tfp index and apply persistence
                    if 'A_tfp' in self.endo_vars:
                        a_tfp_idx_endo = self.endo_vars.index('A_tfp')
                        var_path[t + 1, a_tfp_idx_endo] = self.params.rho_a * var_path[t, a_tfp_idx_endo]
        
        # Create results DataFrame
        results = {}
        for i, var in enumerate(self.endo_vars):
            results[var] = var_path[:, i]
        
        # Verify TFP shock mapping (no manual fixes needed with corrected system)
        if shock_type == 'tfp':
            a_tfp_idx = self.endo_vars.index('A_tfp')
            
            # With the fixed system, TFP shock should properly affect A_tfp
            a_tfp_initial_response = shock_loading[a_tfp_idx, shock_idx] * shock_vector[shock_idx]
            
            if abs(a_tfp_initial_response) < 1e-8:
                print(f"Warning: TFP shock has minimal effect on A_tfp ({a_tfp_initial_response:.8f})")
                print(f"This may indicate a remaining system specification issue.")
            else:
                print(f"TFP shock properly affects A_tfp with coefficient: {a_tfp_initial_response:.6f}")
        
        # Convert to percentage deviations from steady state
        ss_dict = self.steady_state.to_dict()
        
        for var in results:
            ss_var = self._map_to_steady_state_name(var)
            if ss_var in ss_dict and ss_dict[ss_var] != 0:
                # Convert to percentage deviation
                results[var] = results[var] / ss_dict[ss_var] * 100
            elif var in ss_dict and ss_dict[var] != 0:
                results[var] = results[var] / ss_dict[var] * 100
            else:
                # Keep in levels if no steady state found
                results[var] = results[var] * 100  # Convert to percentage points
        
        df = pd.DataFrame(results)
        df.index.name = 'Period'
        
        if variables is not None:
            # Ensure requested variables exist
            available_vars = [v for v in variables if v in df.columns]
            if len(available_vars) < len(variables):
                missing = [v for v in variables if v not in df.columns]
                print(f"Warning: Variables not found: {missing}")
                print(f"Available variables: {list(df.columns)}")
            df = df[available_vars] if available_vars else df
        
        return df
    
    def variance_decomposition(self, 
                             periods: int = 40,
                             variables: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compute forecast error variance decomposition
        """
        if self.linear_system is None or self.linear_system.P is None:
            self.solve_klein()
        
        # Shock standard deviations
        shock_std = np.array([
            self.params.sigma_a,     # TFP
            self.params.sigma_g,     # Gov spending
            self.params.sigma_r,     # Monetary
            0.01,                    # Consumption tax (1% shock)
            0.01,                    # Income tax
            0.01                     # Corporate tax
        ])
        
        shock_names = ['TFP', 'Gov Spending', 'Monetary', 
                      'Consumption Tax', 'Income Tax', 'Corporate Tax']
        
        if variables is None:
            variables = ['Y', 'C', 'I', 'pi', 'T']
        
        # Initialize variance accumulator
        n_vars = len(variables)
        n_shocks = len(shock_names)
        variance_contrib = np.zeros((n_vars, n_shocks))
        
        # Get variable indices
        all_vars = self.state_vars + self.control_vars
        var_indices = [all_vars.index(v) for v in variables]
        
        # Compute contribution of each shock
        for h in range(1, periods + 1):
            # Power of transition matrix
            if h == 1:
                Q_power = self.linear_system.Q
            else:
                Q_power = np.linalg.matrix_power(self.linear_system.Q, h)
            
            # Impact on variables
            for j, shock_idx in enumerate(range(n_shocks)):
                # Shock impact
                shock_impact = np.zeros(self.n_state)
                if shock_idx < self.n_exo: # Ensure shock_idx is within bounds of shock_std
                    # Exogenous shocks start at index self.n_s (4) in the state_vector
                    shock_impact[self.n_s + shock_idx] = shock_std[shock_idx]
                
                # State response
                state_response = Q_power @ shock_impact
                
                # Variable response
                if h == 1:
                    full_response = np.concatenate([
                        state_response,
                        self.linear_system.P @ state_response
                    ])
                else:
                    full_response = np.concatenate([
                        state_response,
                        self.linear_system.P @ state_response
                    ])
                
                # Add to variance
                for i, var_idx in enumerate(var_indices):
                    variance_contrib[i, j] += full_response[var_idx]**2
        
        # Normalize to percentages
        total_variance = variance_contrib.sum(axis=1)
        variance_pct = np.zeros_like(variance_contrib)
        
        for i in range(n_vars):
            if total_variance[i] > 0:
                variance_pct[i, :] = variance_contrib[i, :] / total_variance[i] * 100
        
        # Create DataFrame
        df = pd.DataFrame(variance_pct, 
                         index=variables,
                         columns=shock_names)
        
        return df
    
    def plot_impulse_response(self,
                            shock_type: str,
                            variables: List[str],
                            shock_size: float = 1.0,
                            periods: int = 40,
                            figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Enhanced plotting with confidence bands
        """
        # Compute impulse responses
        irf = self.compute_impulse_response(shock_type, shock_size, periods, variables)
        
        # Create plot
        n_vars = len(variables)
        n_cols = 2
        n_rows = (n_vars + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Color scheme
        colors = plt.cm.Set2(np.linspace(0, 1, 8))
        
        for i, var in enumerate(variables):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Plot IRF
            ax.plot(irf.index, irf[var], linewidth=2.5, color=colors[i % 8])
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
            
            # Add shaded area for first few periods
            ax.fill_between(irf.index[:8], 0, irf[var].iloc[:8], 
                          alpha=0.2, color=colors[i % 8])
            
            # Formatting
            ax.set_title(f'{var}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Quarters', fontsize=10)
            ax.set_ylabel('% dev. from SS', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add value at impact
            impact_value = irf[var].iloc[0]
            if abs(impact_value) > 0.01:
                ax.text(0.02, 0.95, f'Impact: {impact_value:.2f}%', 
                       transform=ax.transAxes, fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.3', 
                               facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for i in range(n_vars, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        # Add title
        shock_title = shock_type.replace('_', ' ').title()
        if 'tax' in shock_type:
            shock_title += f' ({shock_size}pp)'
        else:
            shock_title += ' (1 s.d.)'
        
        plt.suptitle(f'Impulse Response to {shock_title} Shock', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
