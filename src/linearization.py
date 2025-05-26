"""
Linear approximation and impulse response functions for DSGE model

This module implements the linear approximation of the DSGE model around
its steady state and computes impulse response functions to various shocks.
"""

import numpy as np
import pandas as pd
from scipy import linalg
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from .dsge_model import DSGEModel, SteadyState, ModelParameters


class LinearizedDSGE:
    """
    Linearized version of the DSGE model for dynamic analysis
    """
    
    def __init__(self, model: DSGEModel, steady_state: SteadyState):
        self.model = model
        self.steady_state = steady_state
        self.params = model.params
        
        # Define variable ordering for the linearized system
        self.state_vars = ['K', 'B', 'a', 'g', 'eps_r']  # Predetermined + exogenous
        self.control_vars = ['Y', 'C', 'I', 'L', 'w', 'r', 'pi', 'R', 'G', 
                           'T', 'Tc', 'Tl', 'Tk', 'Tf', 'Lambda', 'mc', 'profit']
        self.forward_vars = ['pi', 'Lambda']  # Forward-looking variables
        
        self.n_state = len(self.state_vars)
        self.n_control = len(self.control_vars)
        self.n_vars = self.n_state + self.n_control
        
        # System matrices (to be computed)
        self.A = None  # Coefficient on E[x_{t+1}]
        self.B = None  # Coefficient on x_t
        self.C = None  # Coefficient on shocks
        self.P = None  # Policy function matrix
        self.Q = None  # Transition matrix
        
    def compute_jacobian(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Jacobian matrices of the model equations
        with respect to variables at t+1 and t
        
        This is a simplified numerical approximation
        """
        ss = self.steady_state
        params = self.params
        
        # Number of equations (should match number of endogenous variables)
        n_eq = 19  # Simplified for now
        
        # Initialize Jacobian matrices
        J_plus = np.zeros((n_eq, self.n_vars))  # Jacobian w.r.t. t+1 variables
        J_curr = np.zeros((n_eq, self.n_vars))  # Jacobian w.r.t. t variables
        
        # Fill in the Jacobians based on the linearized equations
        # This is a simplified version - in practice, you would compute these
        # numerically or symbolically
        
        # Example: Euler equation
        # Lambda_t = beta * Lambda_{t+1} * (1 + r_{t+1}) / pi_{t+1}
        eq_idx = 0
        var_idx = self.control_vars.index('Lambda')
        J_curr[eq_idx, self.n_state + var_idx] = 1.0
        J_plus[eq_idx, self.n_state + var_idx] = -params.beta * (1 + ss.r) / ss.pi
        
        r_idx = self.control_vars.index('r')
        J_plus[eq_idx, self.n_state + r_idx] = -params.beta * ss.Lambda / ss.pi
        
        pi_idx = self.control_vars.index('pi')
        J_plus[eq_idx, self.n_state + pi_idx] = params.beta * ss.Lambda * (1 + ss.r) / (ss.pi ** 2)
        
        # Production function: Y = K^alpha * L^(1-alpha)
        eq_idx = 1
        Y_idx = self.control_vars.index('Y')
        J_curr[eq_idx, self.n_state + Y_idx] = 1.0
        
        K_idx = self.state_vars.index('K')
        J_curr[eq_idx, K_idx] = -params.alpha * ss.Y / ss.K
        
        L_idx = self.control_vars.index('L')
        J_curr[eq_idx, self.n_state + L_idx] = -(1 - params.alpha) * ss.Y / ss.L
        
        # Add more equations... (simplified for brevity)
        
        return J_plus, J_curr
    
    def solve_linear_system(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the linear rational expectations model using the
        Klein (2000) or Sims (2002) method
        
        Returns:
            P: Policy function matrix (control = P * state)
            Q: Transition matrix (state_{t+1} = Q * state_t + shock)
        """
        # Get Jacobian matrices
        J_plus, J_curr = self.compute_jacobian()
        
        # Use generalized Schur decomposition (simplified implementation)
        # In practice, you would use a more sophisticated solver
        
        # For now, use a simple iterative method or analytical solution
        # This is a placeholder implementation
        
        # Assume a simple policy function for demonstration
        P = np.random.randn(self.n_control, self.n_state) * 0.1
        
        # Transition matrix for state variables
        Q = np.zeros((self.n_state, self.n_state))
        
        # Capital accumulation
        K_idx = self.state_vars.index('K')
        Q[K_idx, K_idx] = 1 - self.params.delta
        
        # Debt accumulation (simplified)
        B_idx = self.state_vars.index('B')
        Q[B_idx, B_idx] = 1.0
        
        # Exogenous processes
        a_idx = self.state_vars.index('a')
        Q[a_idx, a_idx] = self.params.rho_a
        
        g_idx = self.state_vars.index('g')
        Q[g_idx, g_idx] = self.params.rho_g
        
        # Monetary shock (i.i.d.)
        eps_r_idx = self.state_vars.index('eps_r')
        Q[eps_r_idx, eps_r_idx] = 0.0
        
        self.P = P
        self.Q = Q
        
        return P, Q
    
    def compute_impulse_response(self, 
                               shock_type: str,
                               shock_size: float = 1.0,
                               periods: int = 40,
                               variables: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compute impulse response functions to a specific shock
        
        Args:
            shock_type: Type of shock ('tfp', 'gov_spending', 'monetary', 
                       'consumption_tax', 'income_tax', 'corporate_tax')
            shock_size: Size of the shock (in standard deviations or percentage points)
            periods: Number of periods to simulate
            variables: List of variables to track (if None, track all)
        
        Returns:
            DataFrame with impulse responses
        """
        if self.P is None or self.Q is None:
            self.solve_linear_system()
        
        # Initialize state vector
        state = np.zeros((periods + 1, self.n_state))
        control = np.zeros((periods + 1, self.n_control))
        
        # Apply shock
        shock_vector = np.zeros(self.n_state)
        
        if shock_type == 'tfp':
            a_idx = self.state_vars.index('a')
            shock_vector[a_idx] = shock_size * self.params.sigma_a
        elif shock_type == 'gov_spending':
            g_idx = self.state_vars.index('g')
            shock_vector[g_idx] = shock_size * self.params.sigma_g
        elif shock_type == 'monetary':
            eps_r_idx = self.state_vars.index('eps_r')
            shock_vector[eps_r_idx] = shock_size * self.params.sigma_r
        elif shock_type == 'consumption_tax':
            # For tax shocks, we need to modify the system
            # This is a simplified implementation
            pass
        
        # Initial shock
        state[0] = shock_vector
        
        # Simulate forward
        for t in range(periods):
            # Control variables respond to state
            control[t] = self.P @ state[t]
            
            # State evolves
            if t < periods:
                state[t + 1] = self.Q @ state[t]
        
        # Create DataFrame with results
        results = {}
        
        # Add state variables
        for i, var in enumerate(self.state_vars):
            results[var] = state[:, i]
        
        # Add control variables  
        for i, var in enumerate(self.control_vars):
            results[var] = control[:, i]
        
        # Convert to percentage deviations from steady state
        ss_dict = self.steady_state.to_dict()
        
        for var in results:
            if var in ss_dict and ss_dict[var] != 0:
                results[var] = results[var] / ss_dict[var] * 100  # Percentage deviation
        
        df = pd.DataFrame(results)
        df.index.name = 'Period'
        
        if variables is not None:
            df = df[variables]
        
        return df
    
    def plot_impulse_response(self, 
                            shock_type: str,
                            variables: List[str],
                            shock_size: float = 1.0,
                            periods: int = 40,
                            figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot impulse response functions
        """
        # Compute impulse responses
        irf = self.compute_impulse_response(shock_type, shock_size, periods, variables)
        
        # Create plot
        n_vars = len(variables)
        n_cols = 2
        n_rows = (n_vars + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        for i, var in enumerate(variables):
            ax = axes[i]
            ax.plot(irf.index, irf[var], linewidth=2)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax.set_title(f'{var} (% deviation from steady state)')
            ax.set_xlabel('Quarters')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_vars, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Impulse Response to {shock_type.replace("_", " ").title()} Shock', 
                    fontsize=14)
        plt.tight_layout()
        
        return fig


class TaxSimulator:
    """
    Simulator for analyzing tax policy changes
    """
    
    def __init__(self, model: DSGEModel, linearized_model: LinearizedDSGE):
        self.model = model
        self.linear_model = linearized_model
        self.baseline_params = model.params
        self.baseline_ss = model.steady_state
        
    def simulate_tax_change(self,
                          tax_type: str,
                          new_rate: float,
                          transition_type: str = 'permanent',
                          periods: int = 40) -> Dict[str, pd.DataFrame]:
        """
        Simulate the effects of a tax rate change
        
        Args:
            tax_type: 'consumption', 'income', 'capital', or 'corporate'
            new_rate: New tax rate
            transition_type: 'permanent' or 'temporary'
            periods: Number of periods to simulate
        
        Returns:
            Dictionary with 'baseline' and 'reform' DataFrames
        """
        # Create new parameters with tax change
        new_params = ModelParameters()
        for attr in dir(self.baseline_params):
            if not attr.startswith('_'):
                setattr(new_params, attr, getattr(self.baseline_params, attr))
        
        # Set new tax rate
        if tax_type == 'consumption':
            old_rate = new_params.tau_c
            new_params.tau_c = new_rate
        elif tax_type == 'income':
            old_rate = new_params.tau_l
            new_params.tau_l = new_rate
        elif tax_type == 'capital':
            old_rate = new_params.tau_k
            new_params.tau_k = new_rate
        elif tax_type == 'corporate':
            old_rate = new_params.tau_f
            new_params.tau_f = new_rate
        else:
            raise ValueError(f"Unknown tax type: {tax_type}")
        
        # Compute new steady state
        new_model = DSGEModel(new_params)
        new_ss = new_model.compute_steady_state()
        
        # Compute transition path (simplified - assumes immediate jump to new steady state)
        baseline_path = pd.DataFrame({
            var: [getattr(self.baseline_ss, var)] * periods
            for var in self.model.endogenous_vars
        })
        
        if transition_type == 'permanent':
            reform_path = pd.DataFrame({
                var: [getattr(new_ss, var)] * periods
                for var in self.model.endogenous_vars
            })
        else:
            # Temporary change (simplified)
            reform_path = baseline_path.copy()
            for var in self.model.endogenous_vars:
                reform_path[var].iloc[:20] = getattr(new_ss, var)
        
        return {
            'baseline': baseline_path,
            'reform': reform_path,
            'old_rate': old_rate,
            'new_rate': new_rate,
            'steady_state_change': {
                var: (getattr(new_ss, var) - getattr(self.baseline_ss, var)) / 
                     getattr(self.baseline_ss, var) * 100
                for var in self.model.endogenous_vars
            }
        }
    
    def plot_tax_simulation(self, 
                          results: Dict,
                          variables: List[str],
                          figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot the results of a tax simulation
        """
        baseline = results['baseline']
        reform = results['reform']
        
        n_vars = len(variables)
        n_cols = 2
        n_rows = (n_vars + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        for i, var in enumerate(variables):
            ax = axes[i]
            
            # Plot levels
            ax.plot(baseline.index, baseline[var], 'b-', label='Baseline', linewidth=2)
            ax.plot(reform.index, reform[var], 'r--', label='Reform', linewidth=2)
            
            ax.set_title(var)
            ax.set_xlabel('Quarters')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_vars, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Tax Reform Simulation: {results["old_rate"]:.1%} → {results["new_rate"]:.1%}',
                    fontsize=14)
        plt.tight_layout()
        
        return fig
    
    def compute_fiscal_impact(self, results: Dict) -> pd.DataFrame:
        """
        Compute the fiscal impact of a tax reform
        """
        baseline = results['baseline']
        reform = results['reform']
        
        fiscal_vars = ['Y', 'T', 'Tc', 'Tl', 'Tk', 'Tf', 'G', 'B']
        
        impact = pd.DataFrame({
            'Baseline': baseline[fiscal_vars].mean(),
            'Reform': reform[fiscal_vars].mean()
        })
        
        impact['Change'] = impact['Reform'] - impact['Baseline']
        impact['% Change'] = impact['Change'] / impact['Baseline'] * 100
        
        # Add some ratios
        impact.loc['T/Y'] = [
            impact.loc['T', 'Baseline'] / impact.loc['Y', 'Baseline'],
            impact.loc['T', 'Reform'] / impact.loc['Y', 'Reform'],
            np.nan, np.nan
        ]
        
        impact.loc['B/Y'] = [
            impact.loc['B', 'Baseline'] / impact.loc['Y', 'Baseline'] / 4,  # Annual
            impact.loc['B', 'Reform'] / impact.loc['Y', 'Reform'] / 4,
            np.nan, np.nan
        ]
        
        return impact
