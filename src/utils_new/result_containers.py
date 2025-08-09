"""
Simulation Result Containers

This module provides classes for storing and analyzing simulation results
from DSGE tax policy experiments.

Key Classes:
- SimulationResults: Main container for reform simulation outcomes
- ComparisonResults: Container for multi-scenario comparisons
- WelfareAnalysis: Detailed welfare impact assessment

Economic Variables:
- Y: GDP, C: Consumption, I: Investment, K: Capital, L: Labor
- Welfare measures, fiscal impacts, transition dynamics
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import warnings

# Import dependencies
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from dsge_model import SteadyState
from utils import safe_percentage_change


@dataclass
class SimulationResults:
    """
    Container for tax reform simulation results.
    
    Stores complete simulation output including baseline comparison,
    transition dynamics, welfare analysis, and fiscal impacts.
    
    Attributes:
        name: Descriptive name of the reform scenario
        baseline_path: Time series of baseline scenario (no reform)
        reform_path: Time series with tax reform implemented
        steady_state_baseline: Long-run equilibrium without reform
        steady_state_reform: Long-run equilibrium with reform
        welfare_change: Welfare impact (consumption equivalent)
        fiscal_impact: Government budget effects over time
        transition_periods: Number of periods to reach new steady state
    """
    name: str
    baseline_path: pd.DataFrame
    reform_path: pd.DataFrame
    steady_state_baseline: SteadyState
    steady_state_reform: SteadyState
    welfare_change: float
    fiscal_impact: pd.DataFrame
    transition_periods: int
    
    def __post_init__(self):
        """Validate results after initialization."""
        self._validate_data_consistency()
    
    def _validate_data_consistency(self):
        """Ensure data integrity and consistency."""
        # Check that time series have same length
        if len(self.baseline_path) != len(self.reform_path):
            raise ValueError("Baseline and reform paths must have same length")
        
        # Check for required columns in paths
        # Relax for unit tests that provide minimal variables (e.g., only Y or Y,C)
        required_vars = ['Y', 'C', 'I', 'K', 'L']
        missing_baseline = [v for v in required_vars if v not in self.baseline_path.columns]
        missing_reform = [v for v in required_vars if v not in self.reform_path.columns]
        if missing_baseline or missing_reform:
            warnings.warn(
                f"SimulationResults initialized with missing variables: "
                f"baseline missing {missing_baseline}, reform missing {missing_reform}. "
                f"Analysis methods will operate on available variables only.")
        
        # Validate welfare change is reasonable
        if abs(self.welfare_change) > 0.5:  # 50% welfare change is extreme
            warnings.warn(f"Large welfare change detected: {self.welfare_change:.2%}")
    
    def compute_aggregate_effects(self, variables: List[str], 
                                periods: Optional[int] = None) -> pd.DataFrame:
        """
        Compute aggregate effects over specified periods.
        
        Args:
            variables: List of economic variables to analyze
            periods: Number of periods to average (None = all periods)
            
        Returns:
            DataFrame with baseline, reform, and change statistics
        """
        if periods is None:
            periods = len(self.reform_path)
        
        # Validate requested variables exist
        missing_vars = [v for v in variables if v not in self.baseline_path.columns]
        if missing_vars:
            raise ValueError(f"Variables not found in simulation: {missing_vars}")
        
        effects = {}
        for var in variables:
            baseline_avg = self.baseline_path[var].iloc[:periods].mean()
            reform_avg = self.reform_path[var].iloc[:periods].mean()
            
            effects[var] = {
                'Baseline': baseline_avg,
                'Reform': reform_avg,
                'Change': reform_avg - baseline_avg,
                '% Change': safe_percentage_change(reform_avg, baseline_avg)
            }
        
        return pd.DataFrame(effects).T
    
    def get_impulse_responses(self, variables: List[str], 
                            periods: Optional[int] = None) -> pd.DataFrame:
        """
        Compute impulse response functions (deviations from baseline).
        
        Args:
            variables: Variables to compute IRFs for
            periods: Number of periods (None = all available)
            
        Returns:
            DataFrame with percentage deviations from baseline
        """
        if periods is None:
            periods = len(self.reform_path)
        
        irf_data = {}
        for var in variables:
            if var in self.baseline_path.columns and var in self.reform_path.columns:
                baseline = self.baseline_path[var].iloc[:periods]
                reform = self.reform_path[var].iloc[:periods]
                
                # Compute percentage deviations
                irf_data[var] = ((reform - baseline) / baseline * 100).values
        
        return pd.DataFrame(irf_data, 
                          index=range(periods))
    
    def get_peak_effects(self, variables: List[str]) -> Dict[str, Dict]:
        """
        Find peak effects and when they occur.
        
        Args:
            variables: Variables to analyze
            
        Returns:
            Dictionary with peak magnitudes and timing
        """
        irf = self.get_impulse_responses(variables)
        peak_effects = {}
        
        for var in variables:
            if var in irf.columns:
                series = irf[var]
                
                # Find maximum absolute deviation
                max_idx = series.abs().idxmax()
                peak_value = series.iloc[max_idx]
                
                peak_effects[var] = {
                    'peak_magnitude': peak_value,
                    'peak_period': max_idx,
                    'peak_quarter': f"Q{max_idx + 1}"
                }
        
        return peak_effects
    
    def get_convergence_analysis(self, variables: List[str], 
                               tolerance: float = 0.01) -> Dict[str, int]:
        """
        Analyze convergence to new steady state.
        
        Args:
            variables: Variables to check convergence for
            tolerance: Convergence tolerance (1% = 0.01)
            
        Returns:
            Dictionary with convergence periods for each variable
        """
        irf = self.get_impulse_responses(variables)
        convergence = {}
        
        for var in variables:
            if var in irf.columns:
                series = irf[var].abs()
                
                # Find first period where deviation stays below tolerance
                converged_periods = np.where(series < tolerance * 100)[0]
                
                if len(converged_periods) > 0:
                    # Check for sustained convergence (at least 3 periods)
                    for i, period in enumerate(converged_periods):
                        if i >= 2:  # Need at least 3 consecutive periods
                            if (converged_periods[i] - converged_periods[i-1] == 1 and
                                converged_periods[i-1] - converged_periods[i-2] == 1):
                                convergence[var] = period
                                break
                    else:
                        convergence[var] = converged_periods[0]  # First convergence
                else:
                    convergence[var] = len(irf)  # Hasn't converged
        
        return convergence
    
    def summary_statistics(self) -> Dict:
        """
        Generate comprehensive summary statistics.
        
        Returns:
            Dictionary with key economic impacts and measures
        """
        # Core economic variables
        core_vars = ['Y', 'C', 'I', 'K', 'L']
        available_vars = [v for v in core_vars if v in self.baseline_path.columns]
        
        # Aggregate effects (full simulation period)
        agg_effects = self.compute_aggregate_effects(available_vars)
        
        # Peak effects and timing
        peak_effects = self.get_peak_effects(available_vars)
        
        # Convergence analysis
        convergence = self.get_convergence_analysis(available_vars)
        
        # Steady state comparisons
        steady_state_changes = {}
        for var in available_vars:
            if hasattr(self.steady_state_baseline, var) and hasattr(self.steady_state_reform, var):
                baseline_ss = getattr(self.steady_state_baseline, var)
                reform_ss = getattr(self.steady_state_reform, var)
                steady_state_changes[var] = safe_percentage_change(reform_ss, baseline_ss)
        
        return {
            'reform_name': self.name,
            'welfare_change_percent': self.welfare_change * 100,
            'transition_periods': self.transition_periods,
            'aggregate_effects': agg_effects.to_dict(),
            'peak_effects': peak_effects,
            'convergence_periods': convergence,
            'steady_state_changes_percent': steady_state_changes,
            'simulation_periods': len(self.reform_path)
        }
    
    def to_dict(self) -> Dict:
        """Convert results to dictionary for serialization."""
        return {
            'name': self.name,
            'welfare_change': self.welfare_change,
            'transition_periods': self.transition_periods,
            'baseline_path': self.baseline_path.to_dict(),
            'reform_path': self.reform_path.to_dict(),
            'fiscal_impact': self.fiscal_impact.to_dict(),
            'steady_state_baseline': self.steady_state_baseline.to_dict(),
            'steady_state_reform': self.steady_state_reform.to_dict()
        }


@dataclass  
class ComparisonResults:
    """
    Container for comparing multiple tax reform scenarios.
    
    Enables systematic comparison of different policy options
    with standardized metrics and visualization support.
    """
    scenarios: Dict[str, SimulationResults]
    baseline_name: str = "Baseline"
    
    def get_welfare_ranking(self) -> pd.DataFrame:
        """Rank scenarios by welfare impact."""
        welfare_data = []
        
        for name, results in self.scenarios.items():
            welfare_data.append({
                'Scenario': name,
                'Welfare_Change_Percent': results.welfare_change * 100,
                'Transition_Periods': results.transition_periods
            })
        
        df = pd.DataFrame(welfare_data)
        return df.sort_values('Welfare_Change_Percent', ascending=False)
    
    def get_variable_comparison(self, variable: str, 
                              metric: str = 'avg_change') -> pd.DataFrame:
        """
        Compare specific variable across scenarios.
        
        Args:
            variable: Economic variable to compare
            metric: 'avg_change', 'peak_effect', 'steady_state_change'
        """
        comparison_data = []
        
        for name, results in self.scenarios.items():
            if metric == 'avg_change':
                agg = results.compute_aggregate_effects([variable])
                value = agg.loc[variable, '% Change']
            elif metric == 'peak_effect':
                peaks = results.get_peak_effects([variable])
                value = peaks[variable]['peak_magnitude'] if variable in peaks else None
            elif metric == 'steady_state_change':
                baseline_ss = getattr(results.steady_state_baseline, variable)
                reform_ss = getattr(results.steady_state_reform, variable)
                value = safe_percentage_change(reform_ss, baseline_ss)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            comparison_data.append({
                'Scenario': name,
                'Variable': variable,
                'Metric': metric,
                'Value': value
            })
        
        df = pd.DataFrame(comparison_data)
        return df.pivot(index='Scenario', columns=['Variable', 'Metric'], values='Value')


@dataclass
class WelfareAnalysis:
    """
    Detailed welfare impact analysis for tax reforms.
    
    Provides comprehensive welfare measures including
    consumption equivalents, distributional effects,
    and decomposition by economic channel.
    """
    consumption_equivalent: float
    decomposition: Dict[str, float]
    confidence_interval: Tuple[float, float]
    methodology: str
    
    def get_welfare_summary(self) -> Dict:
        """Generate welfare impact summary."""
        return {
            'consumption_equivalent_percent': self.consumption_equivalent * 100,
            'confidence_lower': self.confidence_interval[0] * 100,
            'confidence_upper': self.confidence_interval[1] * 100,
            'methodology': self.methodology,
            'decomposition': {k: v * 100 for k, v in self.decomposition.items()}
        }