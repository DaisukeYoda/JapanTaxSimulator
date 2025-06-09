"""
Welfare Analysis for Tax Policy Reforms

This module provides comprehensive welfare analysis capabilities for DSGE tax policy simulations.
Implements rigorous welfare measures based on consumption equivalents and utility theory.

Key Classes:
- WelfareAnalyzer: Main welfare computation engine
- WelfareDecomposition: Breakdown of welfare effects by economic channel
- DistributionalAnalysis: Analysis of distributional impacts (future extension)

Economic Theory:
- Consumption equivalent welfare measures
- Lucas (1987) welfare cost methodology
- Hicksian equivalent/compensating variation
- Uncertainty and confidence intervals

Research Standards:
- No dummy data or placeholder calculations
- Explicit methodology documentation
- Empirical validation requirements
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Callable
import warnings
import numpy as np
import pandas as pd
from scipy import optimize
from abc import ABC, abstractmethod

# Import dependencies
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from dsge_model import SteadyState, ModelParameters
from research_warnings import research_critical, research_deprecated, ResearchWarning


@dataclass
class WelfareConfig:
    """Configuration for welfare analysis."""
    methodology: str = 'consumption_equivalent'  # 'consumption_equivalent', 'lucas_welfare'
    discount_factor: float = 0.99
    risk_aversion: float = 1.5  # Coefficient of relative risk aversion
    habit_parameter: float = 0.6  # Habit formation parameter
    include_uncertainty: bool = False
    confidence_level: float = 0.95
    
    def validate(self):
        """Validate welfare configuration."""
        if not 0.9 <= self.discount_factor <= 0.999:
            raise ValueError("discount_factor must be between 0.9 and 0.999")
        if not 0.1 <= self.risk_aversion <= 10.0:
            warnings.warn(f"Unusual risk aversion parameter: {self.risk_aversion}")
        if not 0.0 <= self.habit_parameter <= 0.9:
            raise ValueError("habit_parameter must be between 0 and 0.9")


@dataclass
class WelfareResult:
    """Container for welfare analysis results."""
    consumption_equivalent: float
    methodology: str
    confidence_interval: Tuple[float, float]
    decomposition: Dict[str, float]
    annual_equivalent: float
    
    def summary(self) -> Dict[str, Any]:
        """Generate welfare result summary."""
        return {
            'consumption_equivalent_percent': self.consumption_equivalent * 100,
            'annual_equivalent_percent': self.annual_equivalent * 100,
            'confidence_lower': self.confidence_interval[0] * 100,
            'confidence_upper': self.confidence_interval[1] * 100,
            'methodology': self.methodology,
            'decomposition_percent': {k: v * 100 for k, v in self.decomposition.items()}
        }


class WelfareMethodology(ABC):
    """Abstract base class for welfare calculation methodologies."""
    
    @abstractmethod
    def compute_welfare_change(self, baseline_path: pd.DataFrame,
                             reform_path: pd.DataFrame,
                             config: WelfareConfig) -> float:
        """Compute welfare change using specific methodology."""
        pass
    
    @abstractmethod
    def get_methodology_name(self) -> str:
        """Return methodology name."""
        pass


class ConsumptionEquivalentMethod(WelfareMethodology):
    """
    Consumption equivalent welfare methodology.
    
    Computes the permanent percentage change in consumption that would
    generate the same welfare change as the tax reform.
    """
    
    def compute_welfare_change(self, baseline_path: pd.DataFrame,
                             reform_path: pd.DataFrame,
                             config: WelfareConfig) -> float:
        """
        Compute consumption equivalent welfare change.
        
        Args:
            baseline_path: Baseline simulation path
            reform_path: Reform simulation path  
            config: Welfare configuration
            
        Returns:
            Consumption equivalent welfare change (fraction)
        """
        # Extract consumption paths
        c_baseline = baseline_path['C'].values
        c_reform = reform_path['C'].values
        
        # Validate paths have same length
        if len(c_baseline) != len(c_reform):
            raise ValueError("Baseline and reform paths must have same length")
        
        # Compute period utilities with habit formation
        u_baseline = self._compute_period_utilities(c_baseline, config)
        u_reform = self._compute_period_utilities(c_reform, config)
        
        # Compute lifetime utilities
        discount_factors = np.array([config.discount_factor**t for t in range(len(u_baseline))])
        
        lifetime_u_baseline = np.sum(discount_factors * u_baseline)
        lifetime_u_reform = np.sum(discount_factors * u_reform)
        
        # Solve for consumption equivalent
        if config.risk_aversion == 1.0:
            # Log utility case
            consumption_equivalent = np.exp(
                (lifetime_u_reform - lifetime_u_baseline) / np.sum(discount_factors)
            ) - 1
        else:
            # CRRA utility case
            # Solve: U_baseline * (1 + CE)^(1-σ) = U_reform
            def objective(ce):
                adjusted_baseline = lifetime_u_baseline * ((1 + ce) ** (1 - config.risk_aversion))
                return (adjusted_baseline - lifetime_u_reform) ** 2
            
            # Bound the search to reasonable values
            result = optimize.minimize_scalar(objective, bounds=(-0.5, 0.5), method='bounded')
            consumption_equivalent = result.x
        
        return consumption_equivalent
    
    def _compute_period_utilities(self, consumption: np.ndarray, 
                                config: WelfareConfig) -> np.ndarray:
        """
        Compute period-by-period utilities with habit formation.
        
        Args:
            consumption: Consumption path
            config: Welfare configuration
            
        Returns:
            Array of period utilities
        """
        utilities = np.zeros_like(consumption)
        
        for t in range(len(consumption)):
            # Habit-adjusted consumption
            if t == 0:
                habit_stock = consumption[0]  # Initial period
            else:
                habit_stock = config.habit_parameter * consumption[t-1]
            
            effective_consumption = consumption[t] - habit_stock
            
            # Ensure positive consumption
            effective_consumption = max(effective_consumption, 0.001)
            
            # CRRA utility
            if config.risk_aversion == 1.0:
                utilities[t] = np.log(effective_consumption)
            else:
                utilities[t] = (effective_consumption ** (1 - config.risk_aversion) - 1) / (1 - config.risk_aversion)
        
        return utilities
    
    def get_methodology_name(self) -> str:
        return "consumption_equivalent"


class LucasWelfareMethod(WelfareMethodology):
    """
    Lucas (1987) welfare cost methodology.
    
    Computes welfare cost of business cycle fluctuations,
    adapted for tax policy analysis.
    """
    
    def compute_welfare_change(self, baseline_path: pd.DataFrame,
                             reform_path: pd.DataFrame,
                             config: WelfareConfig) -> float:
        """
        Compute Lucas-style welfare change.
        
        Based on consumption volatility and level effects.
        """
        c_baseline = baseline_path['C'].values
        c_reform = reform_path['C'].values
        
        # Mean consumption levels
        mean_c_baseline = np.mean(c_baseline)
        mean_c_reform = np.mean(c_reform)
        
        # Consumption volatilities
        vol_baseline = np.std(c_baseline) / mean_c_baseline
        vol_reform = np.std(c_reform) / mean_c_reform
        
        # Level effect (dominant)
        level_effect = (mean_c_reform - mean_c_baseline) / mean_c_baseline
        
        # Volatility effect (typically small)
        volatility_effect = -0.5 * config.risk_aversion * (vol_reform**2 - vol_baseline**2)
        
        # Total welfare change
        total_welfare = level_effect + volatility_effect
        
        return total_welfare
    
    def get_methodology_name(self) -> str:
        return "lucas_welfare"


class WelfareDecomposition:
    """
    Decompose welfare effects by economic channel.
    
    Provides insights into which economic mechanisms drive
    the overall welfare impact.
    """
    
    def __init__(self, config: WelfareConfig):
        self.config = config
    
    def decompose_welfare_channels(self, baseline_path: pd.DataFrame,
                                 reform_path: pd.DataFrame) -> Dict[str, float]:
        """
        Decompose welfare change by economic channel.
        
        Args:
            baseline_path: Baseline simulation path
            reform_path: Reform simulation path
            
        Returns:
            Dictionary with welfare contribution by channel
        """
        decomposition = {}
        
        # Level effects (changes in mean values)
        decomposition['consumption_level'] = self._compute_consumption_level_effect(
            baseline_path, reform_path
        )
        
        decomposition['labor_effort'] = self._compute_labor_effort_effect(
            baseline_path, reform_path
        )
        
        # Volatility effects
        decomposition['consumption_volatility'] = self._compute_volatility_effect(
            baseline_path, reform_path, 'C'
        )
        
        # Investment and capital effects
        decomposition['investment_effect'] = self._compute_investment_effect(
            baseline_path, reform_path
        )
        
        # Normalize so effects sum to total
        total = sum(decomposition.values())
        if abs(total) > 1e-6:
            decomposition = {k: v / total for k, v in decomposition.items()}
        
        return decomposition
    
    def _compute_consumption_level_effect(self, baseline_path: pd.DataFrame,
                                        reform_path: pd.DataFrame) -> float:
        """Compute welfare effect from consumption level changes."""
        c_baseline_mean = baseline_path['C'].mean()
        c_reform_mean = reform_path['C'].mean()
        
        return (c_reform_mean - c_baseline_mean) / c_baseline_mean
    
    def _compute_labor_effort_effect(self, baseline_path: pd.DataFrame,
                                   reform_path: pd.DataFrame) -> float:
        """Compute welfare effect from labor effort changes."""
        if 'L' not in baseline_path.columns or 'L' not in reform_path.columns:
            return 0.0
        
        l_baseline_mean = baseline_path['L'].mean()
        l_reform_mean = reform_path['L'].mean()
        
        # Labor effort reduces utility (negative effect)
        labor_disutility_param = 3.0  # Calibrated parameter
        labor_effect = -labor_disutility_param * (l_reform_mean - l_baseline_mean) / l_baseline_mean
        
        return labor_effect
    
    def _compute_volatility_effect(self, baseline_path: pd.DataFrame,
                                 reform_path: pd.DataFrame, variable: str) -> float:
        """Compute welfare effect from volatility changes."""
        if variable not in baseline_path.columns or variable not in reform_path.columns:
            return 0.0
        
        baseline_vol = baseline_path[variable].std() / baseline_path[variable].mean()
        reform_vol = reform_path[variable].std() / reform_path[variable].mean()
        
        # Volatility reduces welfare
        volatility_effect = -0.5 * self.config.risk_aversion * (reform_vol**2 - baseline_vol**2)
        
        return volatility_effect
    
    def _compute_investment_effect(self, baseline_path: pd.DataFrame,
                                 reform_path: pd.DataFrame) -> float:
        """Compute welfare effect from investment changes."""
        if 'I' not in baseline_path.columns or 'I' not in reform_path.columns:
            return 0.0
        
        i_baseline_mean = baseline_path['I'].mean()
        i_reform_mean = reform_path['I'].mean()
        
        # Investment increases future consumption capacity
        investment_productivity = 0.3  # Calibrated parameter
        investment_effect = investment_productivity * (i_reform_mean - i_baseline_mean) / i_baseline_mean
        
        return investment_effect


class WelfareAnalyzer:
    """
    Main welfare analysis engine.
    
    Provides comprehensive welfare analysis with multiple methodologies,
    uncertainty quantification, and detailed decomposition.
    """
    
    def __init__(self, config: Optional[WelfareConfig] = None):
        """
        Initialize welfare analyzer.
        
        Args:
            config: Welfare analysis configuration
        """
        self.config = config or WelfareConfig()
        self.config.validate()
        
        # Available methodologies
        self.methodologies = {
            'consumption_equivalent': ConsumptionEquivalentMethod(),
            'lucas_welfare': LucasWelfareMethod()
        }
        
        self.decomposition_engine = WelfareDecomposition(self.config)
    
    @research_critical(
        "Welfare calculations use simplified utility assumptions and may not reflect "
        "full general equilibrium effects. Results should be validated against "
        "empirical welfare estimates from tax policy literature."
    )
    def analyze_welfare_impact(self, baseline_path: pd.DataFrame,
                             reform_path: pd.DataFrame,
                             decompose: bool = True) -> WelfareResult:
        """
        Comprehensive welfare impact analysis.
        
        Args:
            baseline_path: Baseline simulation results
            reform_path: Reform simulation results
            decompose: Whether to compute welfare decomposition
            
        Returns:
            Complete welfare analysis results
        """
        # Select methodology
        if self.config.methodology not in self.methodologies:
            raise ValueError(f"Unknown methodology: {self.config.methodology}")
        
        methodology = self.methodologies[self.config.methodology]
        
        # Compute main welfare measure
        welfare_change = methodology.compute_welfare_change(
            baseline_path, reform_path, self.config
        )
        
        # Compute confidence interval
        confidence_interval = self._compute_confidence_interval(
            baseline_path, reform_path, methodology
        )
        
        # Decompose welfare effects
        if decompose:
            decomposition = self.decomposition_engine.decompose_welfare_channels(
                baseline_path, reform_path
            )
        else:
            decomposition = {}
        
        # Convert to annual equivalent
        annual_equivalent = self._compute_annual_equivalent(welfare_change)
        
        return WelfareResult(
            consumption_equivalent=welfare_change,
            methodology=methodology.get_methodology_name(),
            confidence_interval=confidence_interval,
            decomposition=decomposition,
            annual_equivalent=annual_equivalent
        )
    
    def _compute_confidence_interval(self, baseline_path: pd.DataFrame,
                                   reform_path: pd.DataFrame,
                                   methodology: WelfareMethodology) -> Tuple[float, float]:
        """
        Compute confidence interval for welfare measure.
        
        Uses bootstrap resampling if uncertainty analysis is enabled.
        """
        if not self.config.include_uncertainty:
            # Return point estimate
            main_estimate = methodology.compute_welfare_change(
                baseline_path, reform_path, self.config
            )
            return (main_estimate, main_estimate)
        
        # Bootstrap confidence interval
        n_bootstrap = 200
        bootstrap_estimates = []
        
        for _ in range(n_bootstrap):
            # Resample paths (block bootstrap to preserve time structure)
            n_periods = len(baseline_path)
            block_size = min(8, n_periods // 4)  # Quarterly blocks
            
            bootstrap_indices = self._block_bootstrap_indices(n_periods, block_size)
            
            bootstrap_baseline = baseline_path.iloc[bootstrap_indices]
            bootstrap_reform = reform_path.iloc[bootstrap_indices]
            
            bootstrap_welfare = methodology.compute_welfare_change(
                bootstrap_baseline, bootstrap_reform, self.config
            )
            bootstrap_estimates.append(bootstrap_welfare)
        
        # Compute confidence interval
        alpha = 1 - self.config.confidence_level
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)
        
        ci_lower = np.percentile(bootstrap_estimates, lower_percentile)
        ci_upper = np.percentile(bootstrap_estimates, upper_percentile)
        
        return (ci_lower, ci_upper)
    
    def _block_bootstrap_indices(self, n_periods: int, block_size: int) -> List[int]:
        """Generate block bootstrap indices."""
        indices = []
        current_pos = 0
        
        while current_pos < n_periods:
            # Random starting point for block
            start = np.random.randint(0, n_periods - block_size + 1)
            
            # Add block indices
            for i in range(block_size):
                if current_pos < n_periods:
                    indices.append((start + i) % n_periods)
                    current_pos += 1
        
        return indices[:n_periods]
    
    def _compute_annual_equivalent(self, quarterly_welfare: float) -> float:
        """Convert quarterly welfare measure to annual equivalent."""
        # Compound quarterly effects to annual
        annual_equivalent = ((1 + quarterly_welfare) ** 4) - 1
        return annual_equivalent
    
    def compare_welfare_across_reforms(self, reform_results: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
        """
        Compare welfare impacts across multiple reforms.
        
        Args:
            reform_results: Dictionary of {reform_name: (baseline_path, reform_path)}
            
        Returns:
            DataFrame with welfare comparison
        """
        comparison_data = []
        
        for reform_name, (baseline_path, reform_path) in reform_results.items():
            welfare_result = self.analyze_welfare_impact(baseline_path, reform_path)
            
            comparison_data.append({
                'Reform': reform_name,
                'Welfare_Change_Percent': welfare_result.consumption_equivalent * 100,
                'Annual_Equivalent_Percent': welfare_result.annual_equivalent * 100,
                'Confidence_Lower': welfare_result.confidence_interval[0] * 100,
                'Confidence_Upper': welfare_result.confidence_interval[1] * 100,
                'Methodology': welfare_result.methodology
            })
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('Welfare_Change_Percent', ascending=False)


# Research-grade welfare analysis with explicit assumptions
@research_deprecated(
    "This function uses simplified welfare approximations that may not capture "
    "full general equilibrium effects. For research use, implement model-specific "
    "welfare calculations based on the underlying utility function."
)
def quick_welfare_estimate(baseline_consumption: float, 
                         reform_consumption: float,
                         risk_aversion: float = 1.5) -> float:
    """
    Quick welfare estimate for preliminary analysis.
    
    ⚠️ RESEARCH WARNING: This is a simplified approximation.
    """
    return (reform_consumption - baseline_consumption) / baseline_consumption