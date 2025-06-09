"""
Tax Reform Definitions

This module provides classes for specifying and managing tax policy reforms
in the DSGE model framework.

Key Classes:
- TaxReform: Specification of tax rate changes and implementation details
- Enhanced reform types for specialized scenarios

Economic Variables:
- tau_c: Consumption tax rate (VAT)
- tau_l: Labor income tax rate  
- tau_k: Capital income tax rate
- tau_f: Corporate tax rate
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
import warnings

# Import dependencies with absolute paths for module isolation
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from dsge_model import ModelParameters


@dataclass
class TaxReform:
    """
    Container for tax reform specification.
    
    This class defines a tax policy change with implementation details
    for simulation in the DSGE model.
    
    Attributes:
        name: Descriptive name for the reform
        tau_c: New consumption tax rate (None = no change)
        tau_l: New labor income tax rate (None = no change)  
        tau_k: New capital income tax rate (None = no change)
        tau_f: New corporate tax rate (None = no change)
        implementation: How the reform is implemented
        duration: Number of periods for temporary reforms
        phase_in_periods: Number of periods for gradual implementation
    
    Implementation Types:
        - 'permanent': Tax change maintained indefinitely
        - 'temporary': Tax change for specified duration, then reverts
        - 'phased': Gradual implementation over multiple periods
    """
    name: str
    tau_c: Optional[float] = None  # New consumption tax rate
    tau_l: Optional[float] = None  # New labor income tax rate
    tau_k: Optional[float] = None  # New capital income tax rate
    tau_f: Optional[float] = None  # New corporate tax rate
    implementation: str = 'permanent'  # 'permanent', 'temporary', 'phased'
    duration: Optional[int] = None  # For temporary reforms
    phase_in_periods: Optional[int] = None  # For phased reforms
    
    def __post_init__(self):
        """Validate reform specification after initialization."""
        self._validate_tax_rates()
        self._validate_implementation_parameters()
    
    def _validate_tax_rates(self):
        """Validate that tax rates are within reasonable bounds."""
        tax_rates = {
            'tau_c': self.tau_c,
            'tau_l': self.tau_l, 
            'tau_k': self.tau_k,
            'tau_f': self.tau_f
        }
        
        for tax_name, rate in tax_rates.items():
            if rate is not None:
                if not (0.0 <= rate <= 0.9):
                    raise ValueError(f"{tax_name}={rate} must be between 0 and 0.9 (90%)")
                
                if rate > 0.6:
                    warnings.warn(f"{tax_name}={rate} is very high (>60%). "
                                "Consider economic feasibility.")
    
    def _validate_implementation_parameters(self):
        """Validate implementation parameters for consistency."""
        if self.implementation == 'temporary':
            if self.duration is None or self.duration <= 0:
                raise ValueError("Temporary reforms require positive duration")
        
        elif self.implementation == 'phased':
            if self.phase_in_periods is None or self.phase_in_periods <= 0:
                raise ValueError("Phased reforms require positive phase_in_periods")
        
        elif self.implementation not in ['permanent', 'temporary', 'phased']:
            raise ValueError(f"Unknown implementation type: {self.implementation}")
    
    def get_changes(self, baseline_params: ModelParameters) -> Dict[str, float]:
        """
        Calculate tax rate changes from baseline parameters.
        
        Args:
            baseline_params: Current model parameters for comparison
            
        Returns:
            Dictionary of tax rate changes (new - baseline)
        """
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
    
    def get_magnitude(self, baseline_params: ModelParameters) -> float:
        """
        Calculate total magnitude of tax changes.
        
        Useful for determining simulation strategy and convergence approach.
        
        Args:
            baseline_params: Current model parameters
            
        Returns:
            Sum of absolute tax rate changes
        """
        changes = self.get_changes(baseline_params)
        return sum(abs(change) for change in changes.values())
    
    def affects_tax(self, tax_type: str) -> bool:
        """
        Check if reform affects a specific tax type.
        
        Args:
            tax_type: One of 'tau_c', 'tau_l', 'tau_k', 'tau_f'
            
        Returns:
            True if the reform changes this tax rate
        """
        return getattr(self, tax_type, None) is not None
    
    def to_dict(self) -> Dict:
        """Convert reform to dictionary for serialization."""
        return {
            'name': self.name,
            'tau_c': self.tau_c,
            'tau_l': self.tau_l,
            'tau_k': self.tau_k,
            'tau_f': self.tau_f,
            'implementation': self.implementation,
            'duration': self.duration,
            'phase_in_periods': self.phase_in_periods
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TaxReform':
        """Create reform from dictionary."""
        return cls(**data)


class SpecializedTaxReforms:
    """
    Factory class for creating common tax reform scenarios.
    
    Provides convenient constructors for typical policy experiments
    used in Japanese tax policy analysis.
    """
    
    @staticmethod
    def consumption_tax_increase(name: str, new_rate: float, 
                               implementation: str = 'permanent') -> TaxReform:
        """
        Create consumption tax increase reform.
        
        Args:
            name: Descriptive name
            new_rate: New consumption tax rate (e.g., 0.10 for 10%)
            implementation: Implementation strategy
        """
        return TaxReform(
            name=name,
            tau_c=new_rate,
            implementation=implementation
        )
    
    @staticmethod
    def income_tax_reduction(name: str, new_rate: float,
                           implementation: str = 'permanent') -> TaxReform:
        """
        Create income tax reduction reform.
        
        Args:
            name: Descriptive name
            new_rate: New labor income tax rate
            implementation: Implementation strategy
        """
        return TaxReform(
            name=name,
            tau_l=new_rate,
            implementation=implementation
        )
    
    @staticmethod
    def revenue_neutral_reform(name: str, tau_c_new: float, 
                             tau_l_new: float) -> TaxReform:
        """
        Create revenue-neutral tax mix reform.
        
        Typically: increase consumption tax, reduce income tax
        
        Args:
            name: Descriptive name
            tau_c_new: New consumption tax rate
            tau_l_new: New labor income tax rate
        """
        return TaxReform(
            name=name,
            tau_c=tau_c_new,
            tau_l=tau_l_new,
            implementation='permanent'
        )
    
    @staticmethod
    def gradual_tax_reform(name: str, tau_c_new: float,
                         phase_periods: int = 8) -> TaxReform:
        """
        Create gradual tax reform (common in Japanese policy).
        
        Args:
            name: Descriptive name  
            tau_c_new: Target consumption tax rate
            phase_periods: Number of quarters for gradual implementation
        """
        return TaxReform(
            name=name,
            tau_c=tau_c_new,
            implementation='phased',
            phase_in_periods=phase_periods
        )


# Common reform scenarios for Japanese tax policy
COMMON_REFORMS = {
    'consumption_tax_10_to_12': SpecializedTaxReforms.consumption_tax_increase(
        "Consumption Tax 10% → 12%", 0.12
    ),
    'consumption_tax_10_to_15': SpecializedTaxReforms.consumption_tax_increase(
        "Consumption Tax 10% → 15%", 0.15
    ),
    'income_tax_reduction_5pp': SpecializedTaxReforms.income_tax_reduction(
        "Income Tax Reduction 5pp", 0.15
    ),
    'revenue_neutral_tax_shift': SpecializedTaxReforms.revenue_neutral_reform(
        "Revenue Neutral: +2pp Consumption, -5pp Income", 0.12, 0.15
    ),
    'gradual_consumption_increase': SpecializedTaxReforms.gradual_tax_reform(
        "Gradual Consumption Tax 10% → 12%", 0.12, 8
    )
}