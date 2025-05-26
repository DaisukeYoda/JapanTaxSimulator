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
from dataclasses import dataclass

from .dsge_model import DSGEModel, SteadyState, ModelParameters


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
    Improved linearization and solution of DSGE models
    """
    
    def __init__(self, model: DSGEModel, steady_state: SteadyState):
        self.model = model
        self.steady_state = steady_state
        self.params = model.params
        
        # Define variable ordering
        # State variables (predetermined and exogenous)
        self.state_vars = ['K', 'B', 'C_lag', 'R_lag', 'a', 'g', 'eps_r', 'tau_c_shock', 'tau_l_shock', 'tau_f_shock']
        
        # Control variables (jump variables)
        self.control_vars = ['Y', 'C', 'I', 'L', 'w', 'r', 'pi', 'R', 'G', 
                           'T', 'Tc', 'Tl', 'Tk', 'Tf', 'Lambda', 'mc', 'profit']
        
        # All endogenous variables
        self.endo_vars = self.state_vars[:4] + self.control_vars  # First 4 state vars are endogenous: K, B, C_lag, R_lag
        
        # Forward-looking variables
        self.forward_vars = ['C', 'pi', 'Lambda'] # R is also forward-looking due to Taylor rule, but not explicitly listed here in some conventions
        
        self.n_state = len(self.state_vars) # Should be 10
        self.n_control = len(self.control_vars) # Should be 17
        self.n_endo = len(self.endo_vars) # Should be 4 + 17 = 21
        self.n_s = 4 # Number of endogenous state variables: K, B, C_lag, R_lag
        self.n_exo = self.n_state - self.n_s  # Exogenous states: a, g, eps_r, and 3 tax shocks. Should be 6.
        
        # Initialize solution matrices
        self.linear_system = None
        
    def build_system_matrices(self) -> LinearizedSystem:
        """
        Build the linearized system matrices A*E[x_{t+1}] + B*x_t + C*z_t = 0
        where x_t contains all endogenous variables and z_t contains exogenous shocks
        """
        ss = self.steady_state
        p = self.params
        
        # Total number of equations
        n_eq = self.n_endo
        
        # Initialize matrices
        A = np.zeros((n_eq, self.n_endo))
        B = np.zeros((n_eq, self.n_endo))
        C = np.zeros((n_eq, self.n_exo))
        
        # Variable indices
        idx = {var: i for i, var in enumerate(self.endo_vars)}
        
        eq = 0  # Equation counter
        
        # 1. Capital accumulation equation
        # K_{t+1} = (1-delta)*K_t + I_t
        A[eq, idx['K']] = 1.0
        B[eq, idx['K']] = -(1 - p.delta)
        B[eq, idx['I']] = -1.0
        eq += 1
        
        # 2. Government debt evolution
        # B_{t+1} = (1+r_t)*B_t + G_t - T_t
        A[eq, idx['B']] = 1.0
        B[eq, idx['B']] = -(1 + ss.r)
        B[eq, idx['r']] = -ss.B
        B[eq, idx['G']] = -1.0
        B[eq, idx['T']] = 1.0
        eq += 1
        
        # 3. Consumption habit formation
        # C_lag_{t+1} = C_t
        A[eq, idx['C_lag']] = 1.0
        B[eq, idx['C']] = -1.0
        eq += 1

        # 4. Law of motion for R_lag
        # R_lag_{t+1} = R_t
        A[eq, idx['R_lag']] = 1.0
        B[eq, idx['R']] = -1.0
        eq += 1
        
        # 5. Euler equation (consumption) - (was 4)
        # Lambda_t = beta * E[Lambda_{t+1} * (1+r_{t+1}) / pi_{t+1}]
        A[eq, idx['Lambda']] = -p.beta * (1 + ss.r) / ss.pi
        A[eq, idx['r']] = -p.beta * ss.Lambda / ss.pi
        A[eq, idx['pi']] = p.beta * ss.Lambda * (1 + ss.r) / (ss.pi**2)
        B[eq, idx['Lambda']] = 1.0
        eq += 1
        
        # 5. Marginal utility of consumption
        # Lambda_t = (C_t - habit*C_lag_t)^(-sigma_c)
        if p.habit > 0:
            denom = (ss.C - p.habit * ss.C)**(-p.sigma_c - 1)
            B[eq, idx['Lambda']] = 1.0
            B[eq, idx['C']] = p.sigma_c * denom
            B[eq, idx['C_lag']] = -p.sigma_c * p.habit * denom
        else:
            B[eq, idx['Lambda']] = 1.0
            B[eq, idx['C']] = p.sigma_c / ss.C
        eq += 1
        
        # 7. Labor supply equation - (was 6)
        # w_t*(1-tau_l)*Lambda_t = chi*L_t^(1/sigma_l)
        B[eq, idx['w']] = (1 - p.tau_l) * ss.Lambda
        B[eq, idx['Lambda']] = (1 - p.tau_l) * ss.w
        B[eq, idx['L']] = -p.chi / p.sigma_l * ss.L**(1/p.sigma_l - 1)
        # Add tax shock effect
        exo_idx = 4  # tau_l_shock index
        C[eq, exo_idx] = -ss.w * ss.Lambda
        eq += 1
        
        # 7. Production function
        # Y_t = A_t * K_t^alpha * L_t^(1-alpha)
        B[eq, idx['Y']] = 1.0
        B[eq, idx['K']] = -p.alpha
        B[eq, idx['L']] = -(1 - p.alpha)
        # TFP shock
        C[eq, 0] = -1.0  # a shock (index for 'a' in C matrix is 0, state_vars index 4)
        eq += 1
        
        # 9. Labor demand (from firm FOC) - (was 8)
        # w_t = mc_t * (1-alpha) * Y_t / L_t
        B[eq, idx['w']] = 1.0
        B[eq, idx['mc']] = -(1 - p.alpha) * ss.Y / ss.L
        B[eq, idx['Y']] = -(1 - p.alpha) * ss.mc / ss.L
        B[eq, idx['L']] = (1 - p.alpha) * ss.mc * ss.Y / (ss.L**2)
        eq += 1
        
        # 9. Capital demand (rental rate)
        # r_t = mc_t * alpha * Y_t / K_t * (1 - tau_f)
        B[eq, idx['r']] = 1.0
        B[eq, idx['mc']] = -p.alpha * ss.Y / ss.K * (1 - p.tau_f)
        B[eq, idx['Y']] = -p.alpha * ss.mc / ss.K * (1 - p.tau_f)
        B[eq, idx['K']] = p.alpha * ss.mc * ss.Y / (ss.K**2) * (1 - p.tau_f)
        # Corporate tax shock
        exo_idx = 5  # tau_f_shock index
        C[eq, exo_idx] = p.alpha * ss.mc * ss.Y / ss.K # exo_idx = 5 for tau_f_shock (state_vars index 9)
        eq += 1
        
        # 11. Phillips curve (New Keynesian) - (was 10)
        # pi_t - pi* = beta*E[pi_{t+1} - pi*] + kappa*mc_t
        kappa = (1 - p.theta_p) * (1 - p.beta * p.theta_p) / p.theta_p
        B[eq, idx['pi']] = 1.0
        A[eq, idx['pi']] = -p.beta
        B[eq, idx['mc']] = -kappa
        eq += 1
        
        # 11. Taylor rule
        # R_t = rho_r*R_{t-1} + (1-rho_r)*[R_ss*(pi_t/pi*)^phi_pi * (Y_t/Y_ss)^phi_y] + eps_r_t
        B[eq, idx['R']] = 1.0
        B[eq, idx['R_lag']] = -p.rho_r # Lagged interest rate term
        B[eq, idx['pi']] = -(1 - p.rho_r) * p.phi_pi
        B[eq, idx['Y']] = -(1 - p.rho_r) * p.phi_y
        # Monetary shock
        C[eq, 2] = -1.0  # eps_r shock (index for 'eps_r' in C matrix is 2, state_vars index 6)
        eq += 1
        
        # 13. Government spending rule - (was 12)
        # G_t = gy_ratio * Y_t * (1 - phi_b*(B_t/Y_t - by_ratio)) + g_shock_t
        B[eq, idx['G']] = 1.0
        B[eq, idx['Y']] = -p.gy_ratio * (1 - p.phi_b * (ss.B / ss.Y - p.by_ratio))
        B[eq, idx['B']] = p.gy_ratio * p.phi_b
        # Government spending shock
        C[eq, 1] = -ss.G  # g shock (index for 'g' in C matrix is 1, state_vars index 5)
        eq += 1
        
        # 14-18. Tax revenue equations - (were 13-17)
        # Consumption tax: Tc_t = tau_c * C_t
        B[eq, idx['Tc']] = 1.0
        B[eq, idx['C']] = -p.tau_c
        C[eq, 3] = -ss.C  # tau_c_shock (index for 'tau_c_shock' in C matrix is 3, state_vars index 7)
        eq += 1
        
        # Labor tax: Tl_t = tau_l * w_t * L_t
        B[eq, idx['Tl']] = 1.0
        B[eq, idx['w']] = -p.tau_l * ss.L
        B[eq, idx['L']] = -p.tau_l * ss.w
        C[eq, 4] = -ss.w * ss.L  # tau_l_shock (index for 'tau_l_shock' in C matrix is 4, state_vars index 8)
        eq += 1
        
        # Capital tax: Tk_t = tau_k * r_t * K_t
        B[eq, idx['Tk']] = 1.0
        B[eq, idx['r']] = -p.tau_k * ss.K
        B[eq, idx['K']] = -p.tau_k * ss.r
        eq += 1
        
        # Corporate tax: Tf_t = tau_f * profit_t
        B[eq, idx['Tf']] = 1.0
        B[eq, idx['profit']] = -p.tau_f
        C[eq, 5] = -ss.profit  # tau_f_shock (index for 'tau_f_shock' in C matrix is 5, state_vars index 9)
        eq += 1
        
        # Total tax: T_t = Tc_t + Tl_t + Tk_t + Tf_t
        B[eq, idx['T']] = 1.0
        B[eq, idx['Tc']] = -1.0
        B[eq, idx['Tl']] = -1.0
        B[eq, idx['Tk']] = -1.0
        B[eq, idx['Tf']] = -1.0
        eq += 1
        
        # 19. Profit equation - (was 18)
        # profit_t = (1 - mc_t) * Y_t
        B[eq, idx['profit']] = 1.0
        B[eq, idx['mc']] = ss.Y
        B[eq, idx['Y']] = -(1 - ss.mc)
        eq += 1
        
        # 19. Goods market clearing
        # Y_t = C_t + I_t + G_t
        B[eq, idx['Y']] = 1.0
        B[eq, idx['C']] = -1.0
        B[eq, idx['I']] = -1.0
        B[eq, idx['G']] = -1.0
        eq += 1
        
        # 21. Fisher equation - (was 20)
        # R_t = (1 + r_t) * pi_{t+1}
        B[eq, idx['R']] = 1.0
        B[eq, idx['r']] = -ss.pi
        A[eq, idx['pi']] = -(1 + ss.r)
        eq += 1
        
        # Store the system
        self.linear_system = LinearizedSystem(
            A=-A,  # Convert to standard form
            B=-B,
            C=-C,
            P=None,
            Q=None,
            R=None,
            var_names=self.endo_vars,
            state_vars=self.state_vars,
            control_vars=self.control_vars
        )
        
        return self.linear_system
    
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
        
        # Generalized Schur decomposition
        T, S, alpha, beta, Q_schur, Z = linalg.ordqz(A, B, sort='ouc')
        
        # Check Blanchard-Kahn conditions
        eigenvalues = alpha / beta
        n_explosive = np.sum(np.abs(eigenvalues) > 1.0)
        n_forward = len(self.forward_vars)
        
        if n_explosive != n_forward:
            print(f"Warning: Blanchard-Kahn conditions not satisfied.")
            print(f"Number of explosive eigenvalues: {n_explosive}")
            print(f"Number of forward-looking variables: {n_forward}")
        
        # Partition the system
        n_s = 4  # Number of endogenous state variables: K, B, C_lag, R_lag
        
        # Extract relevant blocks
        Z11 = Z[:n_s, :n_s]
        Z12 = Z[:n_s, n_s:]
        Z21 = Z[n_s:, :n_s]
        Z22 = Z[n_s:, n_s:]
        
        # Compute policy function
        try:
            P = -linalg.solve(Z22, Z21)
        except:
            print("Warning: Could not solve for policy function, using pseudo-inverse")
            P = -linalg.pinv(Z22) @ Z21
        
        # Compute transition matrix for endogenous states
        Q_endo = Z11 - Z12 @ P
        
        # Build full transition matrix including exogenous processes
        Q_full = np.zeros((self.n_state, self.n_state))
        Q_full[:n_s, :n_s] = Q_endo # Endogenous states part
        
        # Add exogenous process dynamics (indices shifted due to R_lag)
        # state_vars = ['K', 'B', 'C_lag', 'R_lag', 'a', 'g', 'eps_r', 'tau_c_shock', 'tau_l_shock', 'tau_f_shock']
        # Exogenous shocks start at index 4 of state_vars
        Q_full[4, 4] = self.params.rho_a  # TFP persistence ('a' is state_vars[4])
        Q_full[5, 5] = self.params.rho_g  # Gov spending persistence ('g' is state_vars[5])
        # Monetary shock 'eps_r' (state_vars[6]) is i.i.d. (persistence = 0, already zero)
        # Tax shocks (state_vars[7,8,9]) are i.i.d. (persistence = 0, already zero)
        
        # Build full policy matrix
        P_full = np.zeros((self.n_control, self.n_state))
        P_full[:, :n_s] = P[:(self.n_control), :]
        
        # Store solution
        self.linear_system.P = P_full
        self.linear_system.Q = Q_full
        
        # Shock loading matrix
        R_shocks = np.eye(self.n_exo) # Shock matrix for the 6 exogenous shocks
        self.linear_system.R = np.vstack([np.zeros((n_s, self.n_exo)), R_shocks]) # Stack zeros for endogenous states on top
        
        return P_full, Q_full
    
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
        
        # Map shock types to indices
        shock_map = {
            'tfp': 0,
            'gov_spending': 1, 
            'monetary': 2,
            'consumption_tax': 3,
            'income_tax': 4,
            'corporate_tax': 5
        }
        
        if shock_type not in shock_map:
            raise ValueError(f"Unknown shock type: {shock_type}")
        
        shock_idx = shock_map[shock_type]
        
        # Initialize arrays
        state_path = np.zeros((periods + 1, self.n_state))
        control_path = np.zeros((periods + 1, self.n_control))
        
        # Initial shock
        shock_vector = np.zeros(self.n_exo)
        
        # Set shock size based on type
        if shock_type == 'tfp':
            shock_vector[shock_idx] = shock_size * self.params.sigma_a
        elif shock_type == 'gov_spending':
            shock_vector[shock_idx] = shock_size * self.params.sigma_g
        elif shock_type == 'monetary':
            shock_vector[shock_idx] = shock_size * self.params.sigma_r
        elif shock_type in ['consumption_tax', 'income_tax', 'corporate_tax']:
            # For tax shocks, shock_size is in percentage points
            shock_vector[shock_idx] = shock_size / 100  # Convert to decimal
        
        # Apply initial shock
        # Exogenous shocks start at index self.n_s (4) in the state_path vector
        state_path[0, self.n_s:] = shock_vector
        
        # Simulate forward
        for t in range(periods + 1):
            # Control variables respond to state
            control_path[t] = self.linear_system.P @ state_path[t]
            
            # State evolves
            if t < periods:
                state_path[t + 1] = self.linear_system.Q @ state_path[t]
        
        # Create results DataFrame
        results = {}
        
        # Add state variables
        for i, var in enumerate(self.state_vars):
            results[var] = state_path[:, i]
        
        # Add control variables
        for i, var in enumerate(self.control_vars):
            results[var] = control_path[:, i]
        
        # Convert to percentage deviations from steady state
        ss_dict = self.steady_state.to_dict()
        
        for var in results:
            if var in ss_dict and ss_dict[var] != 0:
                # For most variables, compute percentage deviation
                results[var] = results[var] / ss_dict[var] * 100
            elif var in ['a', 'g', 'eps_r', 'tau_c_shock', 'tau_l_shock', 'tau_f_shock', 'R_lag']: # R_lag is a state
                # For shocks and R_lag, keep in levels or specific units
                if var.endswith('_shock'):
                    results[var] = results[var] * 100  # Convert tax/gov/tfp shocks to percentage points if that's the convention
                # R_lag is usually in levels (like R). If it needs conversion, handle here.
                # For now, R_lag will be in same units as R (deviation from SS for R)
                # If R is already % deviation, R_lag will also be.
                # The current IRF code converts R to % dev. from SS if ss.R is non-zero.
                # Let's assume R_lag follows suit.
                # If ss.R_lag is defined and non-zero, it would be: results[var] / ss_dict[var] * 100
                # However, R_lag is not in ss_dict directly. It's ss.R.
                if var == 'R_lag' and 'R' in ss_dict and ss_dict['R'] != 0:
                     results[var] = results[var] / ss_dict['R'] * 100 # Treat R_lag like R for scaling
        
        df = pd.DataFrame(results)
        df.index.name = 'Period'
        
        if variables is not None:
            # Ensure requested variables exist
            available_vars = [v for v in variables if v in df.columns]
            df = df[available_vars]
        
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
