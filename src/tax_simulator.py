"""
Tax Policy Simulator - Backward Compatibility Facade

This module provides a backward compatibility facade that maintains the exact same
interface as the original tax_simulator.py while delegating to the new modular
components underneath.

⚠️ RESEARCH WARNING: This module uses automatic fallbacks that may compromise results.
For research use, import specific modules from simulation/, analysis/, and utils_new/.

Key Classes (Backward Compatible):
- EnhancedTaxSimulator: Main simulator (delegates to new architecture)
- TaxReform: Tax reform specification (imported from utils_new)
- SimulationResults: Results container (imported from utils_new)
- ResearchTaxSimulator: Research-grade simulator (delegates to new architecture)

New modular architecture available in:
- simulation.enhanced_simulator.EnhancedSimulationEngine
- analysis.welfare_analysis.WelfareAnalyzer  
- analysis.fiscal_impact.FiscalAnalyzer
- utils_new.reform_definitions.TaxReform
- utils_new.result_containers.SimulationResults
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
from dataclasses import dataclass

# Import legacy components that are still needed
from .dsge_model import DSGEModel, SteadyState, ModelParameters
from .research_warnings import research_critical, research_deprecated, ResearchWarning

# Import new modular components
from .simulation.enhanced_simulator import EnhancedSimulationEngine, LinearizationConfig
from .simulation.base_simulator import SimulationConfig
from .analysis.welfare_analysis import WelfareAnalyzer, WelfareConfig
from .analysis.fiscal_impact import FiscalAnalyzer, FiscalConfig

# Re-export key classes for backward compatibility
from .utils_new.reform_definitions import TaxReform, SpecializedTaxReforms
from .utils_new.result_containers import SimulationResults, ComparisonResults


class EnhancedTaxSimulator:
    """
    Enhanced tax policy simulator with transition dynamics and welfare analysis.
    
    ⚠️ BACKWARD COMPATIBILITY FACADE: This class maintains the exact same interface
    as the original implementation while using the new modular architecture underneath.
    
    For new development, use:
    - simulation.enhanced_simulator.EnhancedSimulationEngine
    - analysis.welfare_analysis.WelfareAnalyzer
    - analysis.fiscal_impact.FiscalAnalyzer
    """
    
    def __init__(self, 
                 baseline_model: DSGEModel, 
                 use_simple_model: bool = False,
                 use_simple_linearization: Optional[bool] = None,
                 research_mode: bool = False):
        """
        Initialize enhanced tax simulator (backward compatible interface).
        
        Args:
            baseline_model: DSGE model with computed steady state
            use_simple_model: Whether to use simplified model (deprecated)
            use_simple_linearization: Linearization method choice
            research_mode: Enable research-grade validation
        """
        # Store parameters for compatibility
        self.baseline_model = baseline_model
        self.use_simple_model = use_simple_model
        self.use_simple_linearization = use_simple_linearization
        self.research_mode = research_mode
        
        # Create new modular components
        self._setup_modular_components()

        # Expose commonly expected attributes for backward-compat tests
        # linear_model, P_matrix, Q_matrix
        try:
            self.linear_model = self.simulation_engine.linearization_manager.linear_model
            # If Klein succeeded these may exist
            if hasattr(self.linear_model, 'P_matrix'):
                self.P_matrix = self.linear_model.P_matrix
            if hasattr(self.linear_model, 'Q_matrix'):
                self.Q_matrix = self.linear_model.Q_matrix
        except Exception:
            # Keep facade resilient
            pass
        
        # Legacy attributes for backward compatibility
        self.baseline_params = baseline_model.params
        self.baseline_ss = baseline_model.steady_state
        self.results = {}  # For storing results like original
    
    def _setup_modular_components(self):
        """Set up the new modular components."""
        # Simulation configuration
        sim_config = SimulationConfig(
            periods=100,  # Default from original
            validate_results=True,
            compute_welfare=True
        )
        
        # Linearization configuration
        if self.use_simple_linearization is True:
            linearization_method = 'simple'
        elif self.use_simple_linearization is False:
            linearization_method = 'klein'
        else:
            linearization_method = 'auto'
        
        linearization_config = LinearizationConfig(
            method=linearization_method,
            fallback_to_simple=(linearization_method != 'klein')
        )
        
        # Create simulation engine
        self.simulation_engine = EnhancedSimulationEngine(
            baseline_model=self.baseline_model,
            config=sim_config,
            linearization_config=linearization_config,
            research_mode=self.research_mode
        )
        
        # Create analysis components
        self.welfare_analyzer = WelfareAnalyzer(
            config=WelfareConfig(
                methodology='consumption_equivalent',
                include_uncertainty=False
            )
        )
        
        self.fiscal_analyzer = FiscalAnalyzer(
            config=FiscalConfig(
                include_behavioral_responses=True,
                include_general_equilibrium=True
            )
        )
    
    @research_critical(
        "Uses automatic model selection (simple vs complex) with different economic assumptions. "
        "May return results from different underlying models without clear indication. "
        "Welfare calculations use simplified approximations."
    )
    def simulate_reform(self, 
                       reform: TaxReform, 
                       periods: int = 100,
                       compute_welfare: bool = True) -> SimulationResults:
        """
        Simulate a tax reform with full transition dynamics.
        
        This method maintains backward compatibility while using the new
        modular architecture underneath.
        
        Args:
            reform: Tax reform specification
            periods: Number of simulation periods
            compute_welfare: Whether to compute welfare effects
            
        Returns:
            Complete simulation results
        """
        # Use new simulation engine
        results = self.simulation_engine.simulate_reform(reform, periods)
        
        # Enhance with detailed welfare analysis if requested
        if compute_welfare:
            welfare_result = self.welfare_analyzer.analyze_welfare_impact(
                results.baseline_path, results.reform_path
            )
            # Update welfare change with more detailed calculation
            results.welfare_change = welfare_result.consumption_equivalent
        
        # Enhance with detailed fiscal analysis
        fiscal_result = self.fiscal_analyzer.analyze_fiscal_impact(
            results.baseline_path, results.reform_path,
            self.baseline_model.params, 
            self.simulation_engine.create_reform_parameters(reform)
        )
        # Update fiscal impact with detailed analysis
        results.fiscal_impact = fiscal_result.net_fiscal_impact
        
        # Store in legacy results dict for backward compatibility
        self.results[reform.name] = results
        
        return results
    
    def compare_reforms(self, reforms: List[TaxReform], periods: int = 40) -> pd.DataFrame:
        """
        Compare multiple tax reforms (backward compatible interface).
        
        Args:
            reforms: List of tax reforms to compare
            periods: Number of simulation periods
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for reform in reforms:
            print(f"Simulating {reform.name}...")
            results = self.simulate_reform(reform, periods)
            
            comparison_data.append({
                'Reform': reform.name,
                'Welfare_Change_Percent': results.welfare_change * 100,
                'GDP_Change_Percent': self._calculate_gdp_change(results),
                'Revenue_Change': results.fiscal_impact['Net_Impact'].sum(),
                'Implementation': reform.implementation
            })
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('Welfare_Change_Percent', ascending=False)
    
    def _calculate_gdp_change(self, results: SimulationResults) -> float:
        """Calculate average GDP change for backward compatibility."""
        baseline_gdp = results.baseline_path['Y'].mean()
        reform_gdp = results.reform_path['Y'].mean()
        return (reform_gdp - baseline_gdp) / baseline_gdp * 100
    
    # Legacy methods for backward compatibility
    def plot_results(self, results: SimulationResults, **kwargs):
        """Plot simulation results (legacy interface)."""
        warnings.warn(
            "plot_results method deprecated. Use visualization.transition_plots module.",
            DeprecationWarning
        )
        print(f"Plotting {results.name} (legacy method)")
        # Could delegate to visualization module when implemented
    
    def generate_report(self, results: SimulationResults, **kwargs) -> str:
        """Generate text report (legacy interface)."""
        warnings.warn(
            "generate_report method deprecated. Use visualization.report_generation module.",
            DeprecationWarning
        )
        return f"Report for {results.name} (legacy method)"


class ResearchTaxSimulator:
    """
    Research-grade tax simulator (backward compatibility facade).
    
    This class maintains the interface of the original ResearchTaxSimulator
    while using the new modular architecture with research-grade components.
    """
    
    def __init__(self, baseline_model: DSGEModel, use_simple_linearization: bool = False):
        """
        Initialize research-grade simulator.
        
        Args:
            baseline_model: Full DSGE model with computed steady state
            use_simple_linearization: False for full Klein linearization (recommended)
        """
        self.baseline_model = baseline_model
        self.use_simple_linearization = use_simple_linearization
        self.research_mode = True
        
        # Create enhanced simulator with research settings
        self.simulator = EnhancedTaxSimulator(
            baseline_model=baseline_model,
            use_simple_linearization=use_simple_linearization,
            research_mode=True
        )
    
    def simulate_reform(self, reform: TaxReform, periods: int = 40) -> SimulationResults:
        """
        Simulate tax reform with research-grade standards.
        
        Args:
            reform: Tax reform specification
            periods: Number of simulation periods
            
        Returns:
            Complete simulation results with validation
        """
        # Validate reform for research use
        if reform.implementation != 'permanent':
            warnings.warn(
                "Research mode typically uses permanent reforms for policy analysis.",
                ResearchWarning
            )
        
        # Use underlying enhanced simulator
        return self.simulator.simulate_reform(reform, periods, compute_welfare=True)


# Legacy support functions
def load_baseline_model(config_path: str = 'config/parameters.json') -> DSGEModel:
    """Load baseline DSGE model (legacy convenience function)."""
    params = ModelParameters.from_json(config_path)
    model = DSGEModel(params)
    model.steady_state = model.compute_steady_state()
    return model


# Backward compatibility: Re-export common reform scenarios
COMMON_TAX_REFORMS = {
    'consumption_tax_increase_2pp': TaxReform(
        name="Consumption Tax +2pp", tau_c=0.12, implementation='permanent'
    ),
    'income_tax_reduction_5pp': TaxReform(
        name="Income Tax -5pp", tau_l=0.15, implementation='permanent'
    ),
    'revenue_neutral_shift': TaxReform(
        name="Revenue Neutral Shift", tau_c=0.12, tau_l=0.15, implementation='permanent'
    )
}


# Backward compatibility warning
warnings.warn(
    "Using backward compatibility facade. For new development, import directly from:\n"
    "- simulation.enhanced_simulator.EnhancedSimulationEngine\n"
    "- analysis.welfare_analysis.WelfareAnalyzer\n"
    "- analysis.fiscal_impact.FiscalAnalyzer\n"
    "- utils_new.reform_definitions.TaxReform\n"
    "- utils_new.result_containers.SimulationResults",
    FutureWarning,
    stacklevel=2
)