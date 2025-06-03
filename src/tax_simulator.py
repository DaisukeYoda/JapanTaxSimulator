"""
Enhanced Tax Policy Simulator for DSGE Model

This module provides advanced simulation capabilities for analyzing
tax policy changes in the Japanese economy, including transition dynamics,
welfare analysis, and policy optimization.
"""

import numpy as np
import pandas as pd
from scipy import optimize
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json

from .dsge_model import DSGEModel, SteadyState, ModelParameters
from .linearization_improved import ImprovedLinearizedDSGE


@dataclass
class TaxReform:
    """Container for tax reform specification"""
    name: str
    tau_c: Optional[float] = None  # New consumption tax rate
    tau_l: Optional[float] = None  # New labor income tax rate
    tau_k: Optional[float] = None  # New capital income tax rate
    tau_f: Optional[float] = None  # New corporate tax rate
    implementation: str = 'permanent'  # 'permanent', 'temporary', 'phased'
    duration: Optional[int] = None  # For temporary reforms
    phase_in_periods: Optional[int] = None  # For phased reforms
    
    def get_changes(self, baseline_params: ModelParameters) -> Dict[str, float]:
        """Get tax rate changes from baseline"""
        changes = {}
        if self.tau_c is not None:
            changes['tau_c'] = self.tau_c - baseline_params.tau_c
        if self.tau_l is not None:
            changes['tau_l'] = self.tau_l - baseline_params.tau_l
        if self.tau_k is not None:
            changes['tau_k'] = self.tau_k - baseline_params.tau_k
        if self.tau_f is not None:
            changes['tau_f'] = self.tau_f - baseline_params.tau_f
        return changes


@dataclass
class SimulationResults:
    """Container for simulation results"""
    name: str
    baseline_path: pd.DataFrame
    reform_path: pd.DataFrame
    steady_state_baseline: SteadyState
    steady_state_reform: SteadyState
    welfare_change: float
    fiscal_impact: pd.DataFrame
    transition_periods: int
    
    def compute_aggregate_effects(self, variables: List[str], 
                                periods: Optional[int] = None) -> pd.DataFrame:
        """Compute aggregate effects over specified periods"""
        if periods is None:
            periods = len(self.reform_path)
        
        effects = {}
        for var in variables:
            baseline_avg = self.baseline_path[var].iloc[:periods].mean()
            reform_avg = self.reform_path[var].iloc[:periods].mean()
            effects[var] = {
                'Baseline': baseline_avg,
                'Reform': reform_avg,
                'Change': reform_avg - baseline_avg,
                '% Change': (reform_avg - baseline_avg) / baseline_avg * 100
            }
        
        return pd.DataFrame(effects).T


class EnhancedTaxSimulator:
    """
    Enhanced tax policy simulator with transition dynamics and welfare analysis
    """
    
    def __init__(self, baseline_model: DSGEModel):
        self.baseline_model = baseline_model
        self.baseline_params = baseline_model.params
        self.baseline_ss = baseline_model.steady_state
        
        # Create linearized model
        self.linear_model = ImprovedLinearizedDSGE(baseline_model, self.baseline_ss)
        self.linear_model.solve_klein()
        
        # Storage for results
        self.results = {}
        
    def simulate_reform(self, 
                       reform: TaxReform, 
                       periods: int = 100,
                       compute_welfare: bool = True) -> SimulationResults:
        """
        Simulate a tax reform with full transition dynamics
        """
        # Create reform parameters
        reform_params = ModelParameters()
        for attr in dir(self.baseline_params):
            if not attr.startswith('_'):
                setattr(reform_params, attr, getattr(self.baseline_params, attr))
        
        # Apply tax changes
        if reform.tau_c is not None:
            reform_params.tau_c = reform.tau_c
        if reform.tau_l is not None:
            reform_params.tau_l = reform.tau_l
        if reform.tau_k is not None:
            reform_params.tau_k = reform.tau_k
        if reform.tau_f is not None:
            reform_params.tau_f = reform.tau_f
        
        # Compute new steady state using baseline as initial guess
        reform_model = DSGEModel(reform_params)
        reform_ss = reform_model.compute_steady_state(baseline_ss=self.baseline_ss)
        
        # Simulate transition path
        if reform.implementation == 'permanent':
            transition_path = self._simulate_permanent_reform(
                reform.get_changes(self.baseline_params), periods
            )
        elif reform.implementation == 'temporary':
            transition_path = self._simulate_temporary_reform(
                reform.get_changes(self.baseline_params), 
                reform.duration or 20, 
                periods
            )
        elif reform.implementation == 'phased':
            transition_path = self._simulate_phased_reform(
                reform.get_changes(self.baseline_params),
                reform.phase_in_periods or 8,
                periods
            )
        else:
            raise ValueError(f"Unknown implementation type: {reform.implementation}")
        
        # Create baseline path (no reform)
        baseline_path = pd.DataFrame({
            var: [getattr(self.baseline_ss, var)] * periods
            for var in self.baseline_model.endogenous_vars
        })
        baseline_path.index.name = 'Period'
        
        # Compute welfare change if requested
        welfare_change = 0.0
        if compute_welfare:
            welfare_change = self._compute_welfare_change(
                baseline_path, transition_path, self.baseline_params
            )
        
        # Compute fiscal impact
        fiscal_impact = self._compute_fiscal_impact(
            baseline_path, transition_path, periods
        )
        
        # Find transition period (when within 1% of new steady state)
        transition_periods = self._find_transition_period(
            transition_path, reform_ss, tolerance=0.01
        )
        
        # Store and return results
        results = SimulationResults(
            name=reform.name,
            baseline_path=baseline_path,
            reform_path=transition_path,
            steady_state_baseline=self.baseline_ss,
            steady_state_reform=reform_ss,
            welfare_change=welfare_change,
            fiscal_impact=fiscal_impact,
            transition_periods=transition_periods
        )
        
        self.results[reform.name] = results
        return results
    
    def _simulate_permanent_reform(self, 
                                 tax_changes: Dict[str, float],
                                 periods: int) -> pd.DataFrame:
        """Simulate permanent tax reform"""
        # Determine which shocks to use
        shock_sequence = np.zeros((periods, 6))  # 6 types of shocks
        
        if 'tau_c' in tax_changes:
            shock_sequence[:, 3] = tax_changes['tau_c']
        if 'tau_l' in tax_changes:
            shock_sequence[:, 4] = tax_changes['tau_l']
        if 'tau_f' in tax_changes:
            shock_sequence[:, 5] = tax_changes['tau_f']
        
        # Simulate path
        return self._simulate_with_shocks(shock_sequence, periods)
    
    def _simulate_temporary_reform(self,
                                 tax_changes: Dict[str, float],
                                 duration: int,
                                 periods: int) -> pd.DataFrame:
        """Simulate temporary tax reform"""
        shock_sequence = np.zeros((periods, 6))
        
        # Apply shocks only for the duration
        if 'tau_c' in tax_changes:
            shock_sequence[:duration, 3] = tax_changes['tau_c']
        if 'tau_l' in tax_changes:
            shock_sequence[:duration, 4] = tax_changes['tau_l']
        if 'tau_f' in tax_changes:
            shock_sequence[:duration, 5] = tax_changes['tau_f']
        
        return self._simulate_with_shocks(shock_sequence, periods)
    
    def _simulate_phased_reform(self,
                              tax_changes: Dict[str, float],
                              phase_in_periods: int,
                              periods: int) -> pd.DataFrame:
        """Simulate phased-in tax reform"""
        shock_sequence = np.zeros((periods, 6))
        
        # Phase in the changes gradually
        phase_weights = np.linspace(0, 1, phase_in_periods)
        
        if 'tau_c' in tax_changes:
            shock_sequence[:phase_in_periods, 3] = tax_changes['tau_c'] * phase_weights
            shock_sequence[phase_in_periods:, 3] = tax_changes['tau_c']
        if 'tau_l' in tax_changes:
            shock_sequence[:phase_in_periods, 4] = tax_changes['tau_l'] * phase_weights
            shock_sequence[phase_in_periods:, 4] = tax_changes['tau_l']
        if 'tau_f' in tax_changes:
            shock_sequence[:phase_in_periods, 5] = tax_changes['tau_f'] * phase_weights
            shock_sequence[phase_in_periods:, 5] = tax_changes['tau_f']
        
        return self._simulate_with_shocks(shock_sequence, periods)
    
    def _simulate_with_shocks(self, 
                            shock_sequence: np.ndarray,
                            periods: int) -> pd.DataFrame:
        """Simulate model with given shock sequence"""
        # Initialize state and control paths
        state_path = np.zeros((periods, self.linear_model.n_state))
        control_path = np.zeros((periods, self.linear_model.n_control))
        
        # Simulate
        for t in range(periods):
            # Apply shocks
            if t < len(shock_sequence):
                state_path[t, self.linear_model.n_s:] = shock_sequence[t]
            
            # Compute controls
            control_path[t] = self.linear_model.linear_system.P @ state_path[t]
            
            # Update state for next period
            if t < periods - 1:
                state_path[t + 1, :self.linear_model.n_s] = (self.linear_model.linear_system.Q[:self.linear_model.n_s, :] @
                                                            state_path[t])
                if t + 1 < len(shock_sequence):
                    state_path[t + 1, self.linear_model.n_s:] = shock_sequence[t + 1]
        
        # Convert to levels (not deviations)
        results_dict = {}
        ss_dict = self.baseline_ss.to_dict()
        
        # State variables
        for i, var in enumerate(self.linear_model.state_vars[:3]):
            if var in ss_dict:
                results_dict[var] = ss_dict[var] * (1 + state_path[:, i] / 100)
        
        # Control variables
        for i, var in enumerate(self.linear_model.control_vars):
            if var in ss_dict:
                results_dict[var] = ss_dict[var] * (1 + control_path[:, i] / 100)
        
        df = pd.DataFrame(results_dict)
        df.index.name = 'Period'
        
        return df
    
    def _compute_welfare_change(self,
                              baseline_path: pd.DataFrame,
                              reform_path: pd.DataFrame,
                              params: ModelParameters) -> float:
        """
        Compute consumption equivalent welfare change
        """
        # Compute lifetime utility under both scenarios
        periods = len(baseline_path)
        discount_factors = params.beta ** np.arange(periods)
        
        # Utility from consumption and labor
        if params.habit > 0:
            # With habit formation
            C_baseline = baseline_path['C'].values
            C_baseline_lag = np.concatenate([[self.baseline_ss.C], C_baseline[:-1]])
            U_c_baseline = ((C_baseline - params.habit * C_baseline_lag) ** 
                           (1 - params.sigma_c)) / (1 - params.sigma_c)
            
            C_reform = reform_path['C'].values
            C_reform_lag = np.concatenate([[self.baseline_ss.C], C_reform[:-1]])
            U_c_reform = ((C_reform - params.habit * C_reform_lag) ** 
                         (1 - params.sigma_c)) / (1 - params.sigma_c)
        else:
            U_c_baseline = (baseline_path['C'] ** (1 - params.sigma_c)) / (1 - params.sigma_c)
            U_c_reform = (reform_path['C'] ** (1 - params.sigma_c)) / (1 - params.sigma_c)
        
        # Disutility from labor
        U_l_baseline = -params.chi * (baseline_path['L'] ** (1 + 1/params.sigma_l)) / (1 + 1/params.sigma_l)
        U_l_reform = -params.chi * (reform_path['L'] ** (1 + 1/params.sigma_l)) / (1 + 1/params.sigma_l)
        
        # Total discounted utility
        V_baseline = np.sum(discount_factors * (U_c_baseline + U_l_baseline))
        V_reform = np.sum(discount_factors * (U_c_reform + U_l_reform))
        
        # Consumption equivalent variation
        # Find lambda such that V_baseline * (1 + lambda)^(1-sigma_c) = V_reform
        if params.sigma_c == 1:
            # Log utility case
            lambda_ce = np.exp((V_reform - V_baseline) / np.sum(discount_factors)) - 1
        else:
            lambda_ce = ((V_reform / V_baseline) ** (1 / (1 - params.sigma_c))) - 1
        
        return lambda_ce * 100  # Convert to percentage
    
    def _compute_fiscal_impact(self,
                             baseline_path: pd.DataFrame,
                             reform_path: pd.DataFrame,
                             periods: int) -> pd.DataFrame:
        """Compute detailed fiscal impact"""
        fiscal_vars = ['Y', 'T', 'Tc', 'Tl', 'Tk', 'Tf', 'G', 'B']
        
        # Average values over different horizons
        horizons = {
            'Impact (Q1)': 1,
            'Short-run (1 year)': 4,
            'Medium-run (5 years)': 20,
            'Long-run (steady state)': periods
        }
        
        results = {}
        for horizon_name, horizon_periods in horizons.items():
            horizon_results = {}
            for var in fiscal_vars:
                baseline_avg = baseline_path[var].iloc[:horizon_periods].mean()
                reform_avg = reform_path[var].iloc[:horizon_periods].mean()
                horizon_results[var] = {
                    'Baseline': baseline_avg,
                    'Reform': reform_avg,
                    'Change': reform_avg - baseline_avg,
                    '% Change': (reform_avg - baseline_avg) / baseline_avg * 100
                }
            
            # Add fiscal ratios
            baseline_t_y = baseline_path['T'].iloc[:horizon_periods].mean() / baseline_path['Y'].iloc[:horizon_periods].mean()
            reform_t_y = reform_path['T'].iloc[:horizon_periods].mean() / reform_path['Y'].iloc[:horizon_periods].mean()
            
            horizon_results['T/Y ratio'] = {
                'Baseline': baseline_t_y,
                'Reform': reform_t_y,
                'Change': reform_t_y - baseline_t_y,
                '% Change': (reform_t_y - baseline_t_y) / baseline_t_y * 100
            }
            
            results[horizon_name] = horizon_results
        
        # Create multi-index DataFrame
        fiscal_impact = pd.DataFrame(
            {(horizon, var): metrics 
             for horizon, horizon_dict in results.items()
             for var, metrics in horizon_dict.items()}
        ).T
        
        return fiscal_impact
    
    def _find_transition_period(self,
                               path: pd.DataFrame,
                               new_ss: SteadyState,
                               tolerance: float = 0.01) -> int:
        """Find when economy reaches within tolerance of new steady state"""
        key_vars = ['Y', 'C', 'K', 'L']
        
        for t in range(len(path)):
            close_to_ss = True
            for var in key_vars:
                ss_value = getattr(new_ss, var)
                path_value = path[var].iloc[t]
                if abs(path_value - ss_value) / ss_value > tolerance:
                    close_to_ss = False
                    break
            
            if close_to_ss:
                return t
        
        return len(path)  # Didn't converge within simulation period
    
    def compare_reforms(self,
                       reform_list: List[TaxReform],
                       periods: int = 100,
                       variables: Optional[List[str]] = None) -> pd.DataFrame:
        """Compare multiple tax reforms"""
        if variables is None:
            variables = ['Y', 'C', 'I', 'L', 'T', 'T/Y', 'Welfare']
        
        comparison = {}
        
        for reform in reform_list:
            # Simulate reform if not already done
            if reform.name not in self.results:
                results = self.simulate_reform(reform, periods)
            else:
                results = self.results[reform.name]
            
            # Extract key metrics
            reform_metrics = {}
            
            # Steady state changes
            ss_baseline = results.steady_state_baseline
            ss_reform = results.steady_state_reform
            
            for var in variables:
                if var == 'T/Y':
                    baseline_val = ss_baseline.T / ss_baseline.Y
                    reform_val = ss_reform.T / ss_reform.Y
                elif var == 'Welfare':
                    reform_metrics[var] = results.welfare_change
                    continue
                else:
                    baseline_val = getattr(ss_baseline, var)
                    reform_val = getattr(ss_reform, var)
                
                pct_change = (reform_val - baseline_val) / baseline_val * 100
                reform_metrics[var] = pct_change
            
            # Add transition period
            reform_metrics['Transition (quarters)'] = results.transition_periods
            
            comparison[reform.name] = reform_metrics
        
        return pd.DataFrame(comparison).T
    
    def optimal_tax_mix(self,
                       target_revenue: float,
                       tax_bounds: Dict[str, Tuple[float, float]],
                       objective: str = 'welfare') -> Dict[str, float]:
        """
        Find optimal tax mix to achieve target revenue
        
        Args:
            target_revenue: Target tax revenue as share of GDP
            tax_bounds: Bounds for each tax rate
            objective: 'welfare' or 'output'
        """
        
        def objective_function(tax_rates):
            """Objective to minimize (negative welfare or output)"""
            # Create reform
            reform_params = ModelParameters()
            for attr in dir(self.baseline_params):
                if not attr.startswith('_'):
                    setattr(reform_params, attr, getattr(self.baseline_params, attr))
            
            # Set tax rates
            reform_params.tau_c = tax_rates[0]
            reform_params.tau_l = tax_rates[1]
            reform_params.tau_f = tax_rates[2]
            
            try:
                # Compute new steady state
                reform_model = DSGEModel(reform_params)
                reform_ss = reform_model.compute_steady_state()
                
                # Check revenue constraint
                revenue_share = reform_ss.T / reform_ss.Y
                
                if objective == 'welfare':
                    # Approximate welfare change
                    consumption_change = (reform_ss.C - self.baseline_ss.C) / self.baseline_ss.C
                    labor_change = (reform_ss.L - self.baseline_ss.L) / self.baseline_ss.L
                    
                    # Simple welfare approximation
                    welfare = consumption_change - self.baseline_params.chi * labor_change
                    
                    # Penalty for missing revenue target
                    penalty = 100 * (revenue_share - target_revenue) ** 2
                    
                    return -welfare + penalty
                
                elif objective == 'output':
                    output_change = (reform_ss.Y - self.baseline_ss.Y) / self.baseline_ss.Y
                    penalty = 100 * (revenue_share - target_revenue) ** 2
                    return -output_change + penalty
                
            except:
                # Return large penalty if steady state fails
                return 1000.0
        
        # Initial guess
        x0 = [
            self.baseline_params.tau_c,
            self.baseline_params.tau_l,
            self.baseline_params.tau_f
        ]
        
        # Bounds
        bounds = [
            tax_bounds.get('tau_c', (0.0, 0.3)),
            tax_bounds.get('tau_l', (0.0, 0.5)),
            tax_bounds.get('tau_f', (0.0, 0.5))
        ]
        
        # Optimize
        result = optimize.minimize(
            objective_function,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-6, 'maxiter': 100}
        )
        
        if result.success:
            return {
                'tau_c': result.x[0],
                'tau_l': result.x[1],
                'tau_f': result.x[2],
                'objective_value': -result.fun
            }
        else:
            print(f"Optimization failed: {result.message}")
            return None
    
    def plot_transition_dynamics(self,
                               results: SimulationResults,
                               variables: List[str],
                               figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """Plot transition dynamics for a reform"""
        n_vars = len(variables)
        n_cols = 2
        n_rows = (n_vars + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Color palette
        colors = ['#1f77b4', '#ff7f0e']
        
        for i, var in enumerate(variables):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Plot baseline and reform paths
            baseline_values = results.baseline_path[var]
            reform_values = results.reform_path[var]
            
            ax.plot(baseline_values.index, baseline_values, 
                   label='Baseline', color=colors[0], linewidth=2)
            ax.plot(reform_values.index, reform_values,
                   label='Reform', color=colors[1], linewidth=2, linestyle='--')
            
            # Mark steady states
            ax.axhline(y=baseline_values.iloc[-1], color=colors[0], 
                      alpha=0.3, linestyle=':')
            ax.axhline(y=reform_values.iloc[-1], color=colors[1],
                      alpha=0.3, linestyle=':')
            
            # Formatting
            ax.set_title(var, fontsize=12, fontweight='bold')
            ax.set_xlabel('Quarters', fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add percentage change annotation
            pct_change = (reform_values.iloc[-1] - baseline_values.iloc[-1]) / baseline_values.iloc[-1] * 100
            ax.text(0.98, 0.02, f'Δ = {pct_change:+.1f}%',
                   transform=ax.transAxes, fontsize=9,
                   ha='right', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for i in range(n_vars, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.suptitle(f'Transition Dynamics: {results.name}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def generate_report(self, 
                       results: SimulationResults,
                       output_file: str):
        """Generate comprehensive report for a tax reform"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Tax Reform Analysis: {results.name}\n")
            f.write("=" * 60 + "\n\n")
            
            # Summary
            f.write("Executive Summary\n")
            f.write("-" * 30 + "\n")
            f.write(f"Welfare change: {results.welfare_change:+.2f}%\n")
            f.write(f"Transition period: {results.transition_periods} quarters\n")
            
            # Steady state comparison
            f.write("\n\nSteady State Comparison\n")
            f.write("-" * 30 + "\n")
            
            key_vars = ['Y', 'C', 'I', 'L', 'T', 'T/Y', 'B/Y']
            f.write(f"{'Variable':<15} {'Baseline':<12} {'Reform':<12} {'% Change':<12}\n")
            f.write("-" * 51 + "\n")
            
            for var in key_vars:
                if var == 'T/Y':
                    baseline_val = results.steady_state_baseline.T / results.steady_state_baseline.Y
                    reform_val = results.steady_state_reform.T / results.steady_state_reform.Y
                elif var == 'B/Y':
                    baseline_val = results.steady_state_baseline.B / (4 * results.steady_state_baseline.Y)
                    reform_val = results.steady_state_reform.B / (4 * results.steady_state_reform.Y)
                else:
                    baseline_val = getattr(results.steady_state_baseline, var)
                    reform_val = getattr(results.steady_state_reform, var)
                
                pct_change = (reform_val - baseline_val) / baseline_val * 100
                f.write(f"{var:<15} {baseline_val:<12.3f} {reform_val:<12.3f} {pct_change:<+12.2f}\n")
            
            # Fiscal impact
            f.write("\n\nFiscal Impact Analysis\n")
            f.write("-" * 30 + "\n")
            f.write(results.fiscal_impact.to_string())
            
            # Aggregate effects
            f.write("\n\nAggregate Effects (20-year average)\n")
            f.write("-" * 30 + "\n")
            agg_effects = results.compute_aggregate_effects(
                ['Y', 'C', 'I', 'L', 'T'], 
                periods=80  # 20 years
            )
            f.write(agg_effects.to_string())
            
            f.write("\n\nEnd of Report\n")
