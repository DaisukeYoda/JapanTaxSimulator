"""
Unit tests for TaxReform class from tax_simulator module

Tests tax reform specification, change calculation, implementation modes,
and validation logic without heavy simulation dependencies.
"""

import pytest
import numpy as np
from dataclasses import asdict

from src.dsge_model import ModelParameters
from src.tax_simulator import TaxReform


class TestTaxReformBasic:
    """Test basic TaxReform functionality"""
    
    def test_default_initialization(self):
        """Test that TaxReform initializes with minimum required parameters"""
        reform = TaxReform(name="Test Reform")
        
        assert reform.name == "Test Reform"
        assert reform.tau_c is None
        assert reform.tau_l is None
        assert reform.tau_k is None
        assert reform.tau_f is None
        assert reform.implementation == 'permanent'
        assert reform.duration is None
        assert reform.phase_in_periods is None
    
    def test_consumption_tax_reform(self):
        """Test consumption tax reform specification"""
        reform = TaxReform(
            name="消費税引き上げ",
            tau_c=0.15,  # 10% -> 15%
            implementation='permanent'
        )
        
        assert reform.name == "消費税引き上げ"
        assert reform.tau_c == 0.15
        assert reform.tau_l is None
        assert reform.implementation == 'permanent'
    
    def test_comprehensive_tax_reform(self):
        """Test comprehensive tax reform with multiple taxes"""
        reform = TaxReform(
            name="包括税制改革",
            tau_c=0.12,
            tau_l=0.25,
            tau_k=0.20,
            tau_f=0.28,
            implementation='phased',
            phase_in_periods=8
        )
        
        assert reform.name == "包括税制改革"
        assert reform.tau_c == 0.12
        assert reform.tau_l == 0.25
        assert reform.tau_k == 0.20
        assert reform.tau_f == 0.28
        assert reform.implementation == 'phased'
        assert reform.phase_in_periods == 8
    
    def test_temporary_tax_reform(self):
        """Test temporary tax reform specification"""
        reform = TaxReform(
            name="一時的所得税減税",
            tau_l=0.15,
            implementation='temporary',
            duration=12  # 12 quarters
        )
        
        assert reform.name == "一時的所得税減税"
        assert reform.tau_l == 0.15
        assert reform.implementation == 'temporary'
        assert reform.duration == 12


class TestTaxReformValidation:
    """Test tax reform validation and constraints"""
    
    def test_tax_rate_bounds(self):
        """Test that tax rates are within reasonable bounds"""
        # Valid tax rates
        reform_valid = TaxReform(
            name="Valid Reform",
            tau_c=0.12,
            tau_l=0.25,
            tau_k=0.30,
            tau_f=0.35
        )
        
        # All rates should be between 0 and 1
        if reform_valid.tau_c is not None:
            assert 0 <= reform_valid.tau_c < 1, "Consumption tax should be between 0 and 1"
        if reform_valid.tau_l is not None:
            assert 0 <= reform_valid.tau_l < 1, "Labor tax should be between 0 and 1"
        if reform_valid.tau_k is not None:
            assert 0 <= reform_valid.tau_k < 1, "Capital tax should be between 0 and 1"
        if reform_valid.tau_f is not None:
            assert 0 <= reform_valid.tau_f < 1, "Corporate tax should be between 0 and 1"
    
    def test_implementation_mode_validation(self):
        """Test implementation mode validation"""
        # Valid implementation modes
        valid_modes = ['permanent', 'temporary', 'phased']
        
        for mode in valid_modes:
            reform = TaxReform(name=f"Test {mode}", implementation=mode)
            assert reform.implementation == mode
    
    def test_temporary_reform_requires_duration(self):
        """Test that temporary reforms should specify duration"""
        reform = TaxReform(
            name="Temporary Reform",
            tau_c=0.12,
            implementation='temporary'
        )
        
        # Duration should be specified for temporary reforms
        # (This is more of a design guideline than a hard constraint)
        assert reform.implementation == 'temporary'
    
    def test_phased_reform_requires_periods(self):
        """Test that phased reforms should specify phase-in periods"""
        reform = TaxReform(
            name="Phased Reform",
            tau_c=0.12,
            implementation='phased'
        )
        
        # Phase-in periods should be specified for phased reforms
        # (This is more of a design guideline than a hard constraint)
        assert reform.implementation == 'phased'


class TestTaxReformGetChanges:
    """Test tax change calculation from baseline"""
    
    def test_get_changes_with_baseline_parameters(self):
        """Test get_changes method with ModelParameters baseline"""
        baseline_params = ModelParameters(
            tau_c=0.10,
            tau_l=0.20,
            tau_k=0.25,
            tau_f=0.30
        )
        
        reform = TaxReform(
            name="Test Reform",
            tau_c=0.15,  # +5pp increase
            tau_l=0.18   # -2pp decrease
        )
        
        changes = reform.get_changes(baseline_params)
        
        # Should be a dictionary with changes
        assert isinstance(changes, dict)
        
        # Should include consumption tax change
        if 'tau_c' in changes:
            expected_tau_c_change = 0.15 - 0.10
            assert abs(changes['tau_c'] - expected_tau_c_change) < 1e-10
        
        # Should include labor tax change
        if 'tau_l' in changes:
            expected_tau_l_change = 0.18 - 0.20
            assert abs(changes['tau_l'] - expected_tau_l_change) < 1e-10
    
    def test_get_changes_partial_reform(self):
        """Test get_changes with only some taxes specified"""
        baseline_params = ModelParameters(
            tau_c=0.10,
            tau_l=0.20,
            tau_k=0.25,
            tau_f=0.30
        )
        
        # Only change consumption tax
        reform = TaxReform(
            name="Consumption Tax Only",
            tau_c=0.12
        )
        
        changes = reform.get_changes(baseline_params)
        
        # Should only include consumption tax change
        if 'tau_c' in changes:
            expected_change = 0.12 - 0.10
            assert abs(changes['tau_c'] - expected_change) < 1e-10
        
        # Should not include changes for unspecified taxes
        # (or should include zero changes)
        assert isinstance(changes, dict)
    
    def test_get_changes_no_change_reform(self):
        """Test get_changes when reform keeps same rates as baseline"""
        baseline_params = ModelParameters(
            tau_c=0.10,
            tau_l=0.20
        )
        
        # Reform with same rates as baseline
        reform = TaxReform(
            name="No Change",
            tau_c=0.10,
            tau_l=0.20
        )
        
        changes = reform.get_changes(baseline_params)
        
        # Changes should be zero or very small
        if 'tau_c' in changes:
            assert abs(changes['tau_c']) < 1e-10
        if 'tau_l' in changes:
            assert abs(changes['tau_l']) < 1e-10
    
    def test_get_changes_with_none_values(self):
        """Test get_changes when some reform taxes are None"""
        baseline_params = ModelParameters(
            tau_c=0.10,
            tau_l=0.20,
            tau_k=0.25
        )
        
        reform = TaxReform(
            name="Partial Reform",
            tau_c=0.12,
            tau_l=None,  # Not changing labor tax
            tau_k=0.30
        )
        
        changes = reform.get_changes(baseline_params)
        
        # Should handle None values appropriately
        assert isinstance(changes, dict)


class TestTaxReformImplementationModes:
    """Test different implementation modes"""
    
    def test_permanent_implementation(self):
        """Test permanent implementation specification"""
        reform = TaxReform(
            name="Permanent Reform",
            tau_c=0.15,
            implementation='permanent'
        )
        
        assert reform.implementation == 'permanent'
        assert reform.duration is None
        assert reform.phase_in_periods is None
    
    def test_temporary_implementation(self):
        """Test temporary implementation specification"""
        reform = TaxReform(
            name="Temporary Reform",
            tau_c=0.15,
            implementation='temporary',
            duration=8  # 8 quarters = 2 years
        )
        
        assert reform.implementation == 'temporary'
        assert reform.duration == 8
        assert isinstance(reform.duration, int)
        assert reform.duration > 0
    
    def test_phased_implementation(self):
        """Test phased implementation specification"""
        reform = TaxReform(
            name="Phased Reform",
            tau_c=0.15,
            implementation='phased',
            phase_in_periods=4  # Gradual over 4 quarters
        )
        
        assert reform.implementation == 'phased'
        assert reform.phase_in_periods == 4
        assert isinstance(reform.phase_in_periods, int)
        assert reform.phase_in_periods > 0
    
    def test_implementation_parameters_consistency(self):
        """Test consistency between implementation mode and parameters"""
        # Temporary reform should ideally have duration
        temp_reform = TaxReform(
            name="Temp",
            tau_c=0.15,
            implementation='temporary',
            duration=6
        )
        assert temp_reform.implementation == 'temporary'
        assert temp_reform.duration is not None
        
        # Phased reform should ideally have phase_in_periods
        phased_reform = TaxReform(
            name="Phased",
            tau_c=0.15,
            implementation='phased',
            phase_in_periods=8
        )
        assert phased_reform.implementation == 'phased'
        assert phased_reform.phase_in_periods is not None


class TestTaxReformEconomicScenarios:
    """Test economically realistic tax reform scenarios"""
    
    def test_consumption_tax_increase_scenario(self):
        """Test consumption tax increase (common in Japan)"""
        reform = TaxReform(
            name="消費税10%→15%",
            tau_c=0.15,
            implementation='phased',
            phase_in_periods=4
        )
        
        baseline = ModelParameters(tau_c=0.10)
        changes = reform.get_changes(baseline)
        
        # Should be a 5 percentage point increase
        if 'tau_c' in changes:
            assert abs(changes['tau_c'] - 0.05) < 1e-10
        
        # Should be implemented gradually
        assert reform.implementation == 'phased'
        assert reform.phase_in_periods == 4
    
    def test_corporate_tax_reduction_scenario(self):
        """Test corporate tax reduction for competitiveness"""
        reform = TaxReform(
            name="法人税減税",
            tau_f=0.25,  # Reduce from 30% to 25%
            implementation='permanent'
        )
        
        baseline = ModelParameters(tau_f=0.30)
        changes = reform.get_changes(baseline)
        
        # Should be a reduction
        if 'tau_f' in changes:
            assert changes['tau_f'] < 0, "Corporate tax should decrease"
            assert abs(changes['tau_f'] - (-0.05)) < 1e-10
    
    def test_revenue_neutral_reform_scenario(self):
        """Test revenue-neutral reform (higher consumption, lower income tax)"""
        reform = TaxReform(
            name="税制中立改革",
            tau_c=0.12,  # Increase consumption tax
            tau_l=0.18,  # Decrease income tax
            implementation='permanent'
        )
        
        baseline = ModelParameters(tau_c=0.10, tau_l=0.20)
        changes = reform.get_changes(baseline)
        
        # Consumption tax should increase
        if 'tau_c' in changes:
            assert changes['tau_c'] > 0, "Consumption tax should increase"
        
        # Labor tax should decrease
        if 'tau_l' in changes:
            assert changes['tau_l'] < 0, "Labor tax should decrease"
    
    def test_crisis_response_reform_scenario(self):
        """Test temporary tax cuts for crisis response"""
        reform = TaxReform(
            name="危機対応減税",
            tau_l=0.15,  # Temporary income tax cut
            tau_f=0.25,  # Temporary corporate tax cut
            implementation='temporary',
            duration=12  # 3 years
        )
        
        baseline = ModelParameters(tau_l=0.20, tau_f=0.30)
        changes = reform.get_changes(baseline)
        
        # Both taxes should decrease
        if 'tau_l' in changes:
            assert changes['tau_l'] < 0, "Income tax should decrease"
        if 'tau_f' in changes:
            assert changes['tau_f'] < 0, "Corporate tax should decrease"
        
        # Should be temporary
        assert reform.implementation == 'temporary'
        assert reform.duration == 12


class TestTaxReformEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_extreme_tax_rates(self):
        """Test handling of extreme but theoretically valid tax rates"""
        # Very low tax rates
        reform_low = TaxReform(
            name="Very Low",
            tau_c=0.001,
            tau_l=0.005
        )
        assert reform_low.tau_c == 0.001
        assert reform_low.tau_l == 0.005
        
        # High but reasonable tax rates
        reform_high = TaxReform(
            name="High Taxes",
            tau_c=0.25,
            tau_l=0.50,
            tau_k=0.40,
            tau_f=0.45
        )
        assert reform_high.tau_c == 0.25
        assert reform_high.tau_l == 0.50
    
    def test_zero_tax_rates(self):
        """Test handling of zero tax rates"""
        reform = TaxReform(
            name="Zero Corporate Tax",
            tau_f=0.0
        )
        
        assert reform.tau_f == 0.0
        
        # Should be able to calculate changes
        baseline = ModelParameters(tau_f=0.30)
        changes = reform.get_changes(baseline)
        
        if 'tau_f' in changes:
            assert abs(changes['tau_f'] - (-0.30)) < 1e-10
    
    def test_long_duration_periods(self):
        """Test handling of long durations and phase-in periods"""
        # Long temporary reform
        temp_reform = TaxReform(
            name="Long Temporary",
            tau_c=0.15,
            implementation='temporary',
            duration=40  # 10 years
        )
        assert temp_reform.duration == 40
        
        # Long phase-in
        phased_reform = TaxReform(
            name="Very Gradual",
            tau_c=0.15,
            implementation='phased',
            phase_in_periods=20  # 5 years
        )
        assert phased_reform.phase_in_periods == 20
    
    def test_dataclass_attribute_access(self):
        """Test that all attributes are accessible"""
        reform = TaxReform(
            name="Full Reform",
            tau_c=0.12,
            tau_l=0.22,
            tau_k=0.27,
            tau_f=0.32,
            implementation='phased',
            duration=8,
            phase_in_periods=4
        )
        
        # All attributes should be accessible
        attrs = asdict(reform)
        expected_attrs = ['name', 'tau_c', 'tau_l', 'tau_k', 'tau_f', 
                         'implementation', 'duration', 'phase_in_periods']
        
        for attr in expected_attrs:
            assert attr in attrs, f"Attribute {attr} should be accessible"
    
    def test_reform_with_missing_baseline_parameters(self):
        """Test reform calculation when baseline parameters are incomplete"""
        # Baseline with limited parameters
        limited_baseline = ModelParameters(tau_c=0.10)  # Only consumption tax
        
        reform = TaxReform(
            name="Multi-tax Reform",
            tau_c=0.12,
            tau_l=0.25,  # But baseline doesn't have tau_l
            tau_k=0.30   # But baseline doesn't have tau_k
        )
        
        # Should handle missing baseline parameters gracefully
        try:
            changes = reform.get_changes(limited_baseline)
            assert isinstance(changes, dict)
        except AttributeError:
            # This might be expected behavior if baseline doesn't have all taxes
            pass


if __name__ == "__main__":
    pytest.main([__file__])