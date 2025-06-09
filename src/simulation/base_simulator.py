"""
Base Tax Policy Simulation Engine

This module provides the core simulation infrastructure for DSGE tax policy analysis.
Focuses on parameter management, steady state validation, and basic reform orchestration.

Key Classes:
- BaseSimulationEngine: Core simulation infrastructure
- SimulationConfig: Configuration management
- ValidationEngine: Steady state and parameter validation

Economic Variables:
- Y: GDP, C: Consumption, I: Investment, K: Capital, L: Labor
- tau_c: Consumption tax, tau_l: Labor tax, tau_k: Capital tax, tau_f: Corporate tax
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import warnings
import numpy as np
import pandas as pd

# Import dependencies
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from dsge_model import DSGEModel, SteadyState, ModelParameters
from linearization_improved import ImprovedLinearizedDSGE
from utils_new.reform_definitions import TaxReform
from utils_new.result_containers import SimulationResults


@dataclass
class SimulationConfig:
    """
    Configuration for tax policy simulations.
    
    Centralizes simulation parameters and settings for consistent
    behavior across different simulation engines.
    """
    periods: int = 40  # Number of simulation periods
    convergence_tolerance: float = 0.01  # 1% convergence tolerance
    max_steady_state_iterations: int = 1000
    validate_results: bool = True
    compute_welfare: bool = True
    
    # Validation thresholds
    max_gdp_change: float = 0.5  # 50% max GDP change
    max_consumption_change: float = 2.0  # 200% max consumption change
    max_labor_change: float = 0.8  # 80% max labor change
    
    def validate(self):
        """Validate configuration parameters."""
        if self.periods <= 0:
            raise ValueError("periods must be positive")
        if not (0 < self.convergence_tolerance < 1):
            raise ValueError("convergence_tolerance must be between 0 and 1")
        if self.max_steady_state_iterations <= 0:
            raise ValueError("max_steady_state_iterations must be positive")


class ValidationEngine:
    """
    Validation engine for simulation inputs and outputs.
    
    Ensures economic consistency and reasonable parameter bounds
    throughout the simulation process.
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
    
    def validate_reform(self, reform: TaxReform) -> bool:
        """
        Validate tax reform specification.
        
        Args:
            reform: Tax reform to validate
            
        Returns:
            True if reform is valid
            
        Raises:
            ValueError: If reform has invalid parameters
        """
        # Check tax rate bounds
        tax_rates = {
            'tau_c': reform.tau_c,
            'tau_l': reform.tau_l,
            'tau_k': reform.tau_k,
            'tau_f': reform.tau_f
        }
        
        for tax_name, rate in tax_rates.items():
            if rate is not None:
                if not (0.0 <= rate <= 0.9):
                    raise ValueError(f"{tax_name}={rate} must be between 0 and 0.9")
                
                if rate > 0.6:
                    warnings.warn(f"High tax rate: {tax_name}={rate:.1%}")
        
        # Validate implementation parameters
        if reform.implementation == 'temporary' and (reform.duration is None or reform.duration <= 0):
            raise ValueError("Temporary reforms require positive duration")
            
        if reform.implementation == 'phased' and (reform.phase_in_periods is None or reform.phase_in_periods <= 0):
            raise ValueError("Phased reforms require positive phase_in_periods")
        
        return True
    
    def validate_steady_state_change(self, baseline_ss: SteadyState, 
                                   new_ss: SteadyState) -> bool:
        """
        Validate steady state changes are economically reasonable.
        
        Args:
            baseline_ss: Baseline steady state
            new_ss: New steady state after reform
            
        Returns:
            True if changes are within acceptable bounds
        """
        try:
            # Calculate percentage changes
            y_change = abs((new_ss.Y - baseline_ss.Y) / baseline_ss.Y)
            c_change = abs((new_ss.C - baseline_ss.C) / baseline_ss.C)
            l_change = abs((new_ss.L - baseline_ss.L) / baseline_ss.L)
            
            # Check against thresholds
            if y_change > self.config.max_gdp_change:
                warnings.warn(f"Large GDP change: {y_change:.1%}")
                return False
            
            if c_change > self.config.max_consumption_change:
                warnings.warn(f"Large consumption change: {c_change:.1%}")
                return False
            
            if l_change > self.config.max_labor_change:
                warnings.warn(f"Large labor change: {l_change:.1%}")
                return False
            
            # Check for negative values
            if new_ss.Y <= 0 or new_ss.C <= 0 or new_ss.L <= 0:
                warnings.warn("Negative economic variables detected")
                return False
            
            return True
            
        except (AttributeError, ZeroDivisionError, TypeError):
            warnings.warn("Failed to validate steady state changes")
            return False
    
    def validate_simulation_results(self, results: SimulationResults) -> bool:
        """
        Validate complete simulation results.
        
        Args:
            results: Simulation results to validate
            
        Returns:
            True if results pass validation
        """
        # Check data consistency
        if len(results.baseline_path) != len(results.reform_path):
            warnings.warn("Inconsistent path lengths")
            return False
        
        # Check for required variables
        required_vars = ['Y', 'C', 'I', 'K', 'L']
        for var in required_vars:
            if var not in results.baseline_path.columns:
                warnings.warn(f"Missing variable {var} in baseline path")
                return False
            if var not in results.reform_path.columns:
                warnings.warn(f"Missing variable {var} in reform path")
                return False
        
        # Check for extreme welfare changes
        if abs(results.welfare_change) > 0.5:
            warnings.warn(f"Extreme welfare change: {results.welfare_change:.1%}")
        
        # Check for NaN or infinite values
        for path in [results.baseline_path, results.reform_path]:
            if path.isnull().any().any():
                warnings.warn("NaN values detected in simulation paths")
                return False
            
            if np.isinf(path.values).any():
                warnings.warn("Infinite values detected in simulation paths")
                return False
        
        return True


class BaseSimulationEngine(ABC):
    """
    Abstract base class for tax policy simulation engines.
    
    Provides common infrastructure and interface for different
    simulation approaches (simple, enhanced, research-grade).
    """
    
    def __init__(self, baseline_model: DSGEModel, config: Optional[SimulationConfig] = None):
        """
        Initialize base simulation engine.
        
        Args:
            baseline_model: DSGE model with computed steady state
            config: Simulation configuration (None = default)
        """
        self.baseline_model = baseline_model
        self.config = config or SimulationConfig()
        self.config.validate()
        
        self.validator = ValidationEngine(self.config)
        self.results_cache = {}
        
        # Validate baseline model
        self._validate_baseline_model()
    
    def _validate_baseline_model(self):
        """Validate baseline model is properly configured."""
        if self.baseline_model.steady_state is None:
            raise ValueError("Baseline model must have computed steady state")
        
        # Verify essential steady state variables
        ss = self.baseline_model.steady_state
        required_attrs = ['Y', 'C', 'I', 'K', 'L', 'w', 'G', 'B_real']
        
        for attr in required_attrs:
            if not hasattr(ss, attr):
                raise ValueError(f"Steady state missing required attribute: {attr}")
            
            value = getattr(ss, attr)
            if value <= 0:
                warnings.warn(f"Non-positive steady state value: {attr}={value}")
    
    @abstractmethod
    def simulate_reform(self, reform: TaxReform, periods: Optional[int] = None) -> SimulationResults:
        """
        Simulate a tax reform (abstract method).
        
        Args:
            reform: Tax reform specification
            periods: Number of simulation periods (None = use config default)
            
        Returns:
            Complete simulation results
        """
        pass
    
    def create_reform_parameters(self, reform: TaxReform) -> ModelParameters:
        """
        Create model parameters with tax reform applied.
        
        Args:
            reform: Tax reform specification
            
        Returns:
            New model parameters with reform tax rates
        """
        # Validate reform first
        self.validator.validate_reform(reform)
        
        # Create new parameters based on baseline
        new_params = ModelParameters.from_json('config/parameters.json')
        
        # Apply tax changes
        if reform.tau_c is not None:
            new_params.tau_c = reform.tau_c
        if reform.tau_l is not None:
            new_params.tau_l = reform.tau_l
        if reform.tau_k is not None:
            new_params.tau_k = reform.tau_k
        if reform.tau_f is not None:
            new_params.tau_f = reform.tau_f
        
        return new_params
    
    def compute_reform_steady_state(self, reform_params: ModelParameters) -> SteadyState:
        """
        Compute steady state for reform scenario.
        
        Args:
            reform_params: Model parameters with reform applied
            
        Returns:
            New steady state
            
        Raises:
            ValueError: If steady state computation fails
        """
        # Create model with new parameters
        reform_model = DSGEModel(reform_params)
        
        # Compute steady state using baseline as initial guess
        try:
            reform_ss = reform_model.compute_steady_state(
                baseline_ss=self.baseline_model.steady_state
            )
        except Exception as e:
            raise ValueError(f"Failed to compute reform steady state: {e}")
        
        # Validate the new steady state
        if not self.validator.validate_steady_state_change(
            self.baseline_model.steady_state, reform_ss
        ):
            warnings.warn("Reform steady state validation failed")
        
        return reform_ss
    
    def get_cached_result(self, reform: TaxReform) -> Optional[SimulationResults]:
        """
        Retrieve cached simulation result if available.
        
        Args:
            reform: Tax reform specification
            
        Returns:
            Cached results or None if not found
        """
        cache_key = self._generate_cache_key(reform)
        return self.results_cache.get(cache_key)
    
    def cache_result(self, reform: TaxReform, results: SimulationResults):
        """
        Cache simulation results for future use.
        
        Args:
            reform: Tax reform specification
            results: Simulation results to cache
        """
        cache_key = self._generate_cache_key(reform)
        self.results_cache[cache_key] = results
    
    def _generate_cache_key(self, reform: TaxReform) -> str:
        """Generate unique cache key for a reform."""
        reform_dict = reform.to_dict()
        # Create deterministic string representation
        key_parts = [f"{k}:{v}" for k, v in sorted(reform_dict.items()) if v is not None]
        return "|".join(key_parts)
    
    def clear_cache(self):
        """Clear all cached results."""
        self.results_cache.clear()
    
    def get_baseline_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for baseline scenario.
        
        Returns:
            Dictionary with baseline economic indicators
        """
        ss = self.baseline_model.steady_state
        params = self.baseline_model.params
        
        return {
            'gdp': ss.Y,
            'consumption': ss.C,
            'investment': ss.I,
            'capital': ss.K,
            'labor': ss.L,
            'government_spending': ss.G,
            'debt': ss.B_real,
            'tax_rates': {
                'consumption': params.tau_c,
                'labor': params.tau_l,
                'capital': params.tau_k,
                'corporate': params.tau_f
            },
            'ratios': {
                'c_to_y': ss.C / ss.Y,
                'i_to_y': ss.I / ss.Y,
                'g_to_y': ss.G / ss.Y,
                'debt_to_y': ss.B_real / ss.Y
            }
        }