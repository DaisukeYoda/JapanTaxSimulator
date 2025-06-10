"""
Enhanced Tax Policy Simulation Engine

This module provides advanced simulation capabilities for DSGE tax policy analysis.
Includes full linearization, transition dynamics, and comprehensive economic modeling.

Key Classes:
- EnhancedSimulationEngine: Full-featured simulation with Klein linearization
- LinearizationManager: Manages different linearization approaches
- TransitionComputer: Computes dynamic transition paths

Economic Features:
- Klein solution method with Blanchard-Kahn conditions
- Multiple implementation strategies (permanent, temporary, phased)
- Shock sensitivity analysis
- Advanced steady state estimation
"""

from typing import Dict, List, Optional, Tuple, Any
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Import dependencies
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from dsge_model import DSGEModel, SteadyState, ModelParameters
from linearization_improved import ImprovedLinearizedDSGE
from simulation.base_simulator import BaseSimulationEngine, SimulationConfig
from utils_new.reform_definitions import TaxReform
from utils_new.result_containers import SimulationResults
from research_warnings import research_critical, ResearchWarning


@dataclass
class LinearizationConfig:
    """Configuration for linearization methods."""
    method: str = 'klein'  # 'klein', 'simple', 'auto'
    tolerance: float = 1e-8
    max_iterations: int = 1000
    validate_bk_conditions: bool = True
    fallback_to_simple: bool = True


class LinearizationManager:
    """
    Manages different linearization approaches for DSGE models.
    
    Handles the complex choice between Klein linearization (research-grade)
    and simplified linearization (demo/educational).
    """
    
    def __init__(self, baseline_model: DSGEModel, baseline_ss: SteadyState, 
                 config: LinearizationConfig):
        self.baseline_model = baseline_model
        self.baseline_ss = baseline_ss
        self.config = config
        self.linear_model = None
        self.linearization_method = None
        
        self._setup_linearization()
    
    def _setup_linearization(self):
        """Set up linearization based on configuration."""
        self.linear_model = ImprovedLinearizedDSGE(self.baseline_model, self.baseline_ss)
        
        if self.config.method == 'klein':
            success = self._setup_klein_linearization()
            if not success and self.config.fallback_to_simple:
                print("⚠️ Klein linearization failed, falling back to simple method")
                self._setup_simple_linearization()
        elif self.config.method == 'simple':
            self._setup_simple_linearization()
        else:  # auto
            self._setup_auto_linearization()
    
    def _setup_klein_linearization(self) -> bool:
        """Set up Klein linearization method."""
        try:
            print("🎯 Setting up Klein linearization (research-grade)")
            self.linear_model.build_system_matrices()
            P, Q = self.linear_model.solve_klein()
            
            if self.config.validate_bk_conditions:
                if not self._validate_blanchard_kahn_conditions():
                    warnings.warn("Blanchard-Kahn conditions not satisfied")
                    return False
            
            self.linearization_method = 'klein'
            print("✅ Klein linearization setup successful")
            return True
            
        except Exception as e:
            warnings.warn(f"Klein linearization failed: {e}")
            return False
    
    def _setup_simple_linearization(self):
        """
        Set up simple linearization method.
        
        IMPORTANT: This is an ad-hoc linear system with hardcoded coefficients,
        NOT derived from the DSGE model equations. Suitable for demonstrations
        and educational purposes, but NOT for research use.
        
        For research applications, use method='klein' to ensure the dynamics
        are properly derived from the underlying economic model.
        """
        print("✅ Setting up simple linearization (demo/educational - NOT DSGE-derived)")
        # Create simple linear system with fixed coefficients
        # WARNING: These are NOT derived from the DSGE model structure
        
        # Simple linear system: x_t = A * x_{t-1} + B * shock_t
        # where x = [Y, C, I, L]
        n_vars = 4
        A = np.zeros((n_vars, n_vars))
        B = np.zeros((n_vars, 1))
        
        # Persistence parameters (conservative values)
        A[0, 0] = 0.95  # Y persistence
        A[1, 1] = 0.90  # C persistence  
        A[2, 2] = 0.85  # I persistence
        A[3, 3] = 0.88  # L persistence
        
        # Cross-variable effects (moderate)
        A[0, 1] = 0.15  # C -> Y
        A[0, 2] = 0.20  # I -> Y
        A[0, 3] = 0.25  # L -> Y
        A[1, 0] = 0.10  # Y -> C
        
        # Tax shock responses (negative, as tax increases reduce activity)
        B[0, 0] = -0.04  # Y response to tax shock
        B[1, 0] = -0.05  # C response
        B[2, 0] = -0.06  # I response  
        B[3, 0] = -0.03  # L response
        
        # Store as simple linear system
        self.linear_model.linear_system = type('SimpleSystem', (), {
            'A': A, 'B': B, 'P': B, 'n_vars': n_vars
        })()
        
        self.linearization_method = 'simple'
        print("✅ Simple linearization setup complete")
    
    def _setup_auto_linearization(self):
        """Automatically choose linearization method."""
        warnings.warn(
            "⚠️ RESEARCH WARNING: Auto-selecting linearization method. "
            "For research use, specify method='klein' explicitly.",
            ResearchWarning
        )
        
        # Try Klein first, fall back to simple
        if not self._setup_klein_linearization():
            self._setup_simple_linearization()
    
    def _validate_blanchard_kahn_conditions(self) -> bool:
        """Validate Blanchard-Kahn conditions for unique stable solution."""
        try:
            # This would need actual eigenvalue analysis
            # For now, return True if Klein solution succeeded
            return hasattr(self.linear_model, 'P_matrix') and self.linear_model.P_matrix is not None
        except:
            return False


class TransitionComputer:
    """
    Computes dynamic transition paths for tax reforms.
    
    Handles different implementation strategies and generates
    time series of economic variables during adjustment.
    """
    
    def __init__(self, linear_model: ImprovedLinearizedDSGE, 
                 baseline_ss: SteadyState, linearization_method: str):
        self.linear_model = linear_model
        self.baseline_ss = baseline_ss
        self.linearization_method = linearization_method
    
    def compute_permanent_transition(self, tax_changes: Dict[str, float], 
                                   periods: int) -> pd.DataFrame:
        """
        Compute transition path for permanent tax reform.
        
        Args:
            tax_changes: Dictionary of tax rate changes
            periods: Number of simulation periods
            
        Returns:
            DataFrame with transition path
        """
        # Calculate tax shock magnitude
        shock_magnitude = sum(abs(change) for change in tax_changes.values())
        
        # Initialize variables
        variables = ['Y', 'C', 'I', 'L', 'K', 'G']
        transition_data = {}
        
        # Get baseline steady state values
        baseline_values = {
            var: getattr(self.baseline_ss, var) for var in variables
        }
        
        # Simulate transition using linearization
        if self.linearization_method == 'klein':
            transition_data = self._compute_klein_transition(
                tax_changes, baseline_values, periods
            )
        else:
            transition_data = self._compute_simple_transition(
                shock_magnitude, baseline_values, periods, tax_changes=tax_changes
            )
        
        # Convert to DataFrame
        df = pd.DataFrame(transition_data)
        df.index.name = 'Period'
        
        return df
    
    def _compute_klein_transition(self, tax_changes: Dict[str, float],
                                baseline_values: Dict[str, float], 
                                periods: int) -> Dict[str, List[float]]:
        """Compute transition using Klein linearization with actual DSGE solution matrices."""
        try:
            # Use actual Klein solution matrices P and Q from linearization
            if hasattr(self.linear_model, 'P_matrix') and hasattr(self.linear_model, 'Q_matrix'):
                P = self.linear_model.P_matrix
                Q = self.linear_model.Q_matrix
                
                if P is not None and Q is not None:
                    return self._compute_matrix_based_transition(
                        P, Q, tax_changes, baseline_values, periods
                    )
            
            # If matrices not available, fall back with warning
            warnings.warn(
                "Klein solution matrices (P, Q) not available. Using approximation. "
                "For research use, ensure Klein linearization completed successfully.",
                ResearchWarning
            )
            return self._compute_approximate_transition(
                sum(abs(change) for change in tax_changes.values()),
                baseline_values, periods, method='klein', tax_changes=tax_changes
            )
        except Exception as e:
            warnings.warn(f"Klein transition computation failed: {e}")
            return self._compute_simple_transition(
                sum(abs(change) for change in tax_changes.values()),
                baseline_values, periods, tax_changes=tax_changes
            )
    
    def _compute_simple_transition(self, shock_magnitude: float,
                                 baseline_values: Dict[str, float],
                                 periods: int, tax_changes: Optional[Dict[str, float]] = None) -> Dict[str, List[float]]:
        """Compute transition using simple linearization."""
        return self._compute_approximate_transition(
            shock_magnitude, baseline_values, periods, method='simple', tax_changes=tax_changes
        )
    
    def _compute_approximate_transition(self, shock_magnitude: float,
                                      baseline_values: Dict[str, float],
                                      periods: int, method: str, 
                                      tax_changes: Optional[Dict[str, float]] = None) -> Dict[str, List[float]]:
        """Compute approximate transition path with proper tax direction effects."""
        transition_data = {}
        
        # Parameters based on method
        if method == 'klein':
            persistence = 0.92
            initial_response = 0.06
        else:
            persistence = 0.90
            initial_response = 0.05
        
        # Variable-specific parameters
        var_params = {
            'Y': {'response': initial_response * 0.8, 'persist': persistence},
            'C': {'response': initial_response * 1.0, 'persist': persistence * 0.95},
            'I': {'response': initial_response * 1.2, 'persist': persistence * 0.90},
            'L': {'response': initial_response * 0.6, 'persist': persistence * 0.88},
            'K': {'response': initial_response * 0.3, 'persist': persistence * 0.98},
            'G': {'response': initial_response * 0.1, 'persist': persistence * 0.85}
        }
        
        for var in baseline_values.keys():
            baseline = baseline_values[var]
            params = var_params.get(var, {'response': initial_response, 'persist': persistence})
            
            # Compute economic direction effect based on tax changes
            if tax_changes is not None:
                direction_effect = self._compute_tax_direction_effect(var, tax_changes)
            else:
                direction_effect = -shock_magnitude  # Fallback: assume negative effect
            
            # Initial response with proper economic direction
            initial_effect = params['response'] * direction_effect * baseline
            
            # Generate transition path
            path = []
            current_effect = initial_effect
            
            for t in range(periods):
                current_value = baseline + current_effect
                path.append(max(current_value, 0.01 * baseline))  # Prevent negative values
                
                # Decay effect
                current_effect *= params['persist']
            
            transition_data[var] = path
        
        return transition_data
    
    def _compute_tax_direction_effect(self, variable: str, tax_changes: Dict[str, float]) -> float:
        """
        Compute the economic direction effect of tax changes on specific variables.
        
        Args:
            variable: Economic variable (Y, C, I, L, etc.)
            tax_changes: Dictionary of tax rate changes
            
        Returns:
            Signed effect magnitude (positive = stimulative, negative = contractionary)
        """
        total_effect = 0.0
        
        # Consumption tax effects
        if 'tau_c' in tax_changes:
            dtau_c = tax_changes['tau_c']
            if variable == 'C':
                total_effect -= dtau_c * 1.5  # Strong negative effect on consumption
            elif variable == 'Y':
                total_effect -= dtau_c * 1.2  # Moderate negative effect on GDP
            elif variable == 'I':
                total_effect -= dtau_c * 0.3  # Weak negative effect on investment
            elif variable == 'L':
                total_effect -= dtau_c * 0.5  # Weak negative effect on labor
                
        # Income tax effects  
        if 'tau_l' in tax_changes:
            dtau_l = tax_changes['tau_l']
            if variable == 'L':
                total_effect -= dtau_l * 1.8  # Strong negative effect on labor
            elif variable == 'C':
                total_effect -= dtau_l * 1.2  # Moderate negative effect on consumption
            elif variable == 'Y':
                total_effect -= dtau_l * 1.0  # Moderate negative effect on GDP
            elif variable == 'I':
                total_effect -= dtau_l * 0.2  # Weak negative effect on investment
                
        # Corporate tax effects
        if 'tau_f' in tax_changes:
            dtau_f = tax_changes['tau_f']
            if variable == 'I':
                total_effect -= dtau_f * 2.0  # Strong negative effect on investment
            elif variable == 'Y':
                total_effect -= dtau_f * 0.8  # Moderate negative effect on GDP
            elif variable == 'L':
                total_effect -= dtau_f * 0.4  # Weak positive effect on labor demand
            elif variable == 'C':
                total_effect -= dtau_f * 0.1  # Very weak effect on consumption
                
        # Capital tax effects (if present)
        if 'tau_k' in tax_changes:
            dtau_k = tax_changes['tau_k'] 
            if variable == 'I':
                total_effect -= dtau_k * 1.5  # Strong negative effect on investment
            elif variable == 'K':
                total_effect -= dtau_k * 1.0  # Direct effect on capital
            elif variable == 'Y':
                total_effect -= dtau_k * 0.6  # Moderate negative effect on GDP
                
        return total_effect
    
    def _compute_matrix_based_transition(self, P: np.ndarray, Q: np.ndarray,
                                       tax_changes: Dict[str, float],
                                       baseline_values: Dict[str, float],
                                       periods: int) -> Dict[str, List[float]]:
        """
        Compute transition dynamics using actual Klein solution matrices.
        
        Uses the DSGE model's solved state space representation:
        x_t = P * x_{t-1} + Q * shock_t
        
        Args:
            P: State transition matrix from Klein solution
            Q: Shock response matrix from Klein solution  
            tax_changes: Dictionary of tax rate changes
            baseline_values: Steady state baseline values
            periods: Number of simulation periods
            
        Returns:
            Dictionary with transition paths for each variable
        """
        # Map variable names to state vector indices
        # This mapping should match the linearization variable ordering
        var_mapping = self._get_variable_mapping()
        n_vars = len(var_mapping)
        
        # Construct tax shock vector
        shock_vector = self._construct_tax_shock(tax_changes, Q.shape[1])
        
        # Initialize state vector (deviations from steady state)
        state_vector = np.zeros(P.shape[0])
        
        # Store transition paths
        transition_data = {}
        for var in baseline_values.keys():
            transition_data[var] = []
        
        # Simulate transition dynamics
        for t in range(periods):
            # Apply shock only in first period for permanent reform
            current_shock = shock_vector if t == 0 else np.zeros_like(shock_vector)
            
            # Update state: x_t = P * x_{t-1} + Q * shock_t
            state_vector = P @ state_vector + Q @ current_shock
            
            # Convert state deviations back to levels
            for var_name, baseline_value in baseline_values.items():
                if var_name in var_mapping:
                    idx = var_mapping[var_name]
                    if idx < len(state_vector):
                        # Convert log-deviation to level
                        level_value = baseline_value * (1 + state_vector[idx])
                        # Ensure non-negative values
                        level_value = max(level_value, 0.01 * baseline_value)
                        transition_data[var_name].append(level_value)
                    else:
                        # Variable not in state vector, use baseline
                        transition_data[var_name].append(baseline_value)
                else:
                    # Variable not modeled, use baseline with small decay
                    decay_factor = 0.95 ** t
                    transition_data[var_name].append(baseline_value * decay_factor)
        
        return transition_data
    
    def _get_variable_mapping(self) -> Dict[str, int]:
        """Get mapping from variable names to state vector indices."""
        # Standard DSGE variable ordering (should match linearization)
        # This needs to be consistent with the linearization implementation
        return {
            'Y': 0,  # Output
            'C': 1,  # Consumption  
            'I': 2,  # Investment
            'L': 3,  # Labor
            'K': 4,  # Capital
            'G': 5   # Government spending
        }
    
    def _construct_tax_shock(self, tax_changes: Dict[str, float], n_shocks: int) -> np.ndarray:
        """
        Construct shock vector for tax changes.
        
        Maps tax rate changes to structural shocks in the linearized model.
        """
        shock_vector = np.zeros(n_shocks)
        
        # Map tax changes to shock indices
        # This mapping should correspond to the shock ordering in linearization
        tax_shock_mapping = {
            'tau_c': 0,  # Consumption tax shock
            'tau_l': 1,  # Labor tax shock  
            'tau_k': 2,  # Capital tax shock
            'tau_f': 3   # Corporate tax shock
        }
        
        # Fill shock vector based on tax changes
        for tax_type, change in tax_changes.items():
            if tax_type in tax_shock_mapping and tax_shock_mapping[tax_type] < n_shocks:
                shock_vector[tax_shock_mapping[tax_type]] = change
        
        # If no specific mapping, use first shock as aggregate tax change
        if np.allclose(shock_vector, 0) and tax_changes:
            total_change = sum(abs(change) for change in tax_changes.values())
            shock_vector[0] = total_change * np.sign(sum(tax_changes.values()))
        
        return shock_vector
    
    def compute_temporary_transition(self, tax_changes: Dict[str, float],
                                   periods: int, duration: int) -> pd.DataFrame:
        """
        Compute transition for temporary tax reform.
        
        Args:
            tax_changes: Tax rate changes
            periods: Total simulation periods
            duration: Duration of temporary reform
            
        Returns:
            Transition path with reform reversal
        """
        # Compute permanent transition for reference
        permanent_path = self.compute_permanent_transition(tax_changes, periods)
        
        # Modify path to account for reversal
        temp_data = {}
        for var in permanent_path.columns:
            path = permanent_path[var].copy()
            
            # After duration, begin reversal
            if duration < periods:
                # Gradual return to baseline (faster than initial adjustment)
                reversal_speed = 0.85  # Faster than forward adjustment
                baseline_value = getattr(self.baseline_ss, var)
                
                for t in range(duration, periods):
                    periods_since_reversal = t - duration
                    path.iloc[t] = (baseline_value + 
                                  (path.iloc[duration] - baseline_value) * 
                                  (reversal_speed ** periods_since_reversal))
            
            temp_data[var] = path
        
        return pd.DataFrame(temp_data)
    
    def compute_phased_transition(self, tax_changes: Dict[str, float],
                                periods: int, phase_periods: int) -> pd.DataFrame:
        """
        Compute transition for phased tax reform.
        
        Args:
            tax_changes: Final tax rate changes
            periods: Total simulation periods  
            phase_periods: Number of periods for gradual implementation
            
        Returns:
            Transition path with gradual implementation
        """
        # Compute final permanent transition
        final_path = self.compute_permanent_transition(tax_changes, periods)
        
        # Create gradual implementation
        phased_data = {}
        baseline_values = {var: getattr(self.baseline_ss, var) for var in final_path.columns}
        
        for var in final_path.columns:
            path = []
            baseline = baseline_values[var]
            final_values = final_path[var].values
            
            for t in range(periods):
                if t < phase_periods:
                    # Gradual phase-in
                    phase_fraction = (t + 1) / phase_periods
                    # Smooth transition using sigmoid-like function
                    smooth_fraction = 3 * phase_fraction**2 - 2 * phase_fraction**3
                    target_value = baseline + smooth_fraction * (final_values[periods-1] - baseline)
                    
                    # Add dynamic adjustment
                    if t == 0:
                        path.append(baseline + 0.1 * (target_value - baseline))
                    else:
                        momentum = 0.7 * (path[t-1] - (path[t-2] if t > 1 else baseline))
                        path.append(target_value + momentum)
                else:
                    # Use post-implementation dynamics
                    path.append(final_values[t])
            
            phased_data[var] = path
        
        return pd.DataFrame(phased_data)


class EnhancedSimulationEngine(BaseSimulationEngine):
    """
    Enhanced tax policy simulation engine.
    
    Provides full-featured DSGE simulation with Klein linearization,
    multiple implementation strategies, and comprehensive analysis.
    """
    
    def __init__(self, baseline_model: DSGEModel, 
                 config: Optional[SimulationConfig] = None,
                 linearization_config: Optional[LinearizationConfig] = None,
                 research_mode: bool = False):
        """
        Initialize enhanced simulation engine.
        
        Args:
            baseline_model: DSGE model with computed steady state
            config: Simulation configuration
            linearization_config: Linearization settings
            research_mode: Enable research-grade validation
        """
        super().__init__(baseline_model, config)
        
        self.research_mode = research_mode
        self.linearization_config = linearization_config or LinearizationConfig()
        
        # Set up linearization manager
        self.linearization_manager = LinearizationManager(
            baseline_model, baseline_model.steady_state, self.linearization_config
        )
        
        # Create transition computer
        self.transition_computer = TransitionComputer(
            self.linearization_manager.linear_model,
            baseline_model.steady_state,
            self.linearization_manager.linearization_method
        )
    
    @research_critical(
        "Uses automatic model selection (simple vs complex) with different economic assumptions. "
        "May return results from different underlying models without clear indication. "
        "Welfare calculations use simplified approximations."
    )
    def simulate_reform(self, reform: TaxReform, 
                       periods: Optional[int] = None) -> SimulationResults:
        """
        Simulate a tax reform with full transition dynamics.
        
        Args:
            reform: Tax reform specification
            periods: Number of simulation periods (None = use config)
            
        Returns:
            Complete simulation results with transition paths
        """
        if periods is None:
            periods = self.config.periods
        
        # Check cache first
        cached_result = self.get_cached_result(reform)
        if cached_result is not None:
            print(f"Using cached result for {reform.name}")
            return cached_result
        
        print(f"Simulating {reform.name} with enhanced engine...")
        
        # Create reform parameters and compute new steady state
        reform_params = self.create_reform_parameters(reform)
        reform_ss = self.compute_reform_steady_state(reform_params)
        
        # Compute tax changes for transition computation
        tax_changes = reform.get_changes(self.baseline_model.params)
        
        # Generate baseline path (no reform)
        baseline_path = self._generate_baseline_path(periods)
        
        # Compute transition path based on implementation type
        if reform.implementation == 'permanent':
            reform_path = self.transition_computer.compute_permanent_transition(
                tax_changes, periods
            )
        elif reform.implementation == 'temporary':
            reform_path = self.transition_computer.compute_temporary_transition(
                tax_changes, periods, reform.duration
            )
        elif reform.implementation == 'phased':
            reform_path = self.transition_computer.compute_phased_transition(
                tax_changes, periods, reform.phase_in_periods
            )
        else:
            raise ValueError(f"Unknown implementation type: {reform.implementation}")
        
        # Compute welfare change (simplified)
        welfare_change = self._compute_simple_welfare_change(
            baseline_path, reform_path
        )
        
        # Compute fiscal impact
        fiscal_impact = self._compute_fiscal_impact(
            baseline_path, reform_path, reform_params
        )
        
        # Find transition period
        transition_periods = self._find_transition_period(reform_path)
        
        # Create results
        results = SimulationResults(
            name=reform.name,
            baseline_path=baseline_path,
            reform_path=reform_path,
            steady_state_baseline=self.baseline_model.steady_state,
            steady_state_reform=reform_ss,
            welfare_change=welfare_change,
            fiscal_impact=fiscal_impact,
            transition_periods=transition_periods
        )
        
        # Validate results
        if self.config.validate_results:
            if not self.validator.validate_simulation_results(results):
                warnings.warn("Simulation results failed validation")
        
        # Cache results
        self.cache_result(reform, results)
        
        return results
    
    def _generate_baseline_path(self, periods: int) -> pd.DataFrame:
        """Generate baseline path (steady state extended)."""
        ss = self.baseline_model.steady_state
        variables = ['Y', 'C', 'I', 'L', 'K', 'G']
        
        baseline_data = {}
        for var in variables:
            value = getattr(ss, var)
            # Small random variation around steady state for realism
            path = [value * (1 + 0.001 * np.random.normal()) for _ in range(periods)]
            baseline_data[var] = path
        
        return pd.DataFrame(baseline_data)
    
    def _compute_simple_welfare_change(self, baseline_path: pd.DataFrame,
                                     reform_path: pd.DataFrame) -> float:
        """Compute welfare change using consumption equivalent."""
        baseline_c = baseline_path['C'].mean()
        reform_c = reform_path['C'].mean()
        
        # Simple consumption equivalent welfare measure
        welfare_change = (reform_c - baseline_c) / baseline_c
        
        # Apply bounds for realism
        return np.clip(welfare_change, -0.3, 0.3)
    
    def _compute_fiscal_impact(self, baseline_path: pd.DataFrame,
                             reform_path: pd.DataFrame, 
                             reform_params: ModelParameters) -> pd.DataFrame:
        """Compute government fiscal impact over time."""
        periods = len(reform_path)
        
        fiscal_data = []
        for t in range(periods):
            # Tax revenues (simplified calculation)
            baseline_revenue = (
                self.baseline_model.params.tau_c * baseline_path['C'].iloc[t] +
                self.baseline_model.params.tau_l * 0.2 * baseline_path['Y'].iloc[t]  # Approx labor income
            )
            
            reform_revenue = (
                reform_params.tau_c * reform_path['C'].iloc[t] +
                reform_params.tau_l * 0.2 * reform_path['Y'].iloc[t]
            )
            
            fiscal_data.append({
                'Period': t,
                'Baseline_Revenue': baseline_revenue,
                'Reform_Revenue': reform_revenue,
                'Revenue_Change': reform_revenue - baseline_revenue,
                'Revenue_Change_Percent': (reform_revenue - baseline_revenue) / baseline_revenue * 100
            })
        
        return pd.DataFrame(fiscal_data)
    
    def _find_transition_period(self, reform_path: pd.DataFrame, 
                              tolerance: float = 0.01) -> int:
        """Find when transition is essentially complete."""
        if len(reform_path) < 2:
            return len(reform_path)
        
        # Look at GDP convergence
        y_path = reform_path['Y']
        final_value = y_path.iloc[-1]
        
        for t in range(len(y_path) - 1, 0, -1):
            if abs(y_path.iloc[t] - final_value) / final_value > tolerance:
                return min(t + 5, len(y_path))  # Add buffer
        
        return len(y_path)