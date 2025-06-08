"""
Unit tests for ModelParameters class from dsge_model module

Tests parameter initialization, validation, JSON loading, and edge cases.
"""

import pytest
import numpy as np
import json
import tempfile
import os
from dataclasses import asdict

from src.dsge_model import ModelParameters


class TestModelParametersBasic:
    """Test basic ModelParameters functionality"""
    
    def test_default_initialization(self):
        """Test that ModelParameters initializes with default values"""
        params = ModelParameters()
        
        # Test household parameters
        assert params.beta == 0.99
        assert params.sigma_c == 1.5
        assert params.sigma_l == 2.0
        assert params.habit == 0.6
        assert params.chi == 3.0
        
        # Test firm parameters
        assert params.alpha == 0.33
        assert params.delta == 0.025
        assert params.theta_p == 0.75
        assert params.epsilon == 6.0
        assert params.psi == 4.0
        
        # Test government parameters
        assert params.gy_ratio == 0.20
        assert params.by_ratio == 2.0
        assert params.rho_g == 0.9
        assert params.phi_b == 0.1
        assert params.tau_l_ss == 0.20
        assert params.tau_l == 0.20
        
        # Test monetary policy parameters
        assert params.phi_pi == 1.5
        assert params.phi_y == 0.125
        assert params.rho_r == 0.8
        assert params.pi_target == 1.005
        
        # Test tax parameters
        assert params.tau_c == 0.10
        assert params.tau_k == 0.25
        assert params.tau_f == 0.30
    
    def test_custom_initialization(self):
        """Test ModelParameters initialization with custom values"""
        params = ModelParameters(
            beta=0.95,
            sigma_c=2.0,
            tau_c=0.15,
            tau_l=0.25
        )
        
        assert params.beta == 0.95
        assert params.sigma_c == 2.0
        assert params.tau_c == 0.15
        assert params.tau_l == 0.25
        
        # Other parameters should remain at defaults
        assert params.sigma_l == 2.0
        assert params.alpha == 0.33


class TestModelParametersValidation:
    """Test parameter validation and economic constraints"""
    
    def test_beta_bounds(self):
        """Test that beta is within economically reasonable bounds"""
        params = ModelParameters()
        assert 0 < params.beta < 1, "Beta should be between 0 and 1 for finite value"
    
    def test_positive_parameters(self):
        """Test that parameters that should be positive are positive"""
        params = ModelParameters()
        
        positive_params = [
            'sigma_c', 'sigma_l', 'habit', 'chi', 'alpha', 'delta', 
            'theta_p', 'epsilon', 'psi', 'gy_ratio', 'by_ratio',
            'phi_pi', 'phi_y', 'rho_r', 'pi_target'
        ]
        
        for param_name in positive_params:
            value = getattr(params, param_name)
            assert value > 0, f"{param_name} should be positive, got {value}"
    
    def test_tax_rate_bounds(self):
        """Test that tax rates are within reasonable bounds"""
        params = ModelParameters()
        
        tax_rates = ['tau_c', 'tau_l', 'tau_k', 'tau_f', 'tau_l_ss']
        
        for tax_rate in tax_rates:
            value = getattr(params, tax_rate)
            assert 0 <= value < 1, f"{tax_rate} should be between 0 and 1, got {value}"
    
    def test_shock_persistence(self):
        """Test that shock persistence parameters are within unit circle"""
        params = ModelParameters()
        
        persistence_params = ['rho_a', 'rho_g', 'rho_r', 'rho_ystar']
        
        for param_name in persistence_params:
            value = getattr(params, param_name)
            assert 0 <= value < 1, f"{param_name} should be between 0 and 1 for stationarity, got {value}"
    
    def test_taylor_principle(self):
        """Test that monetary policy satisfies Taylor principle"""
        params = ModelParameters()
        assert params.phi_pi > 1, "Taylor principle requires phi_pi > 1 for determinacy"


class TestModelParametersJsonLoading:
    """Test JSON loading functionality"""
    
    def create_test_json(self, content):
        """Helper to create temporary JSON file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(content, f)
            return f.name
    
    def test_from_json_basic(self):
        """Test basic JSON loading"""
        test_data = {
            "model_parameters": {
                "household": {
                    "beta": 0.98,
                    "sigma_c": 1.8,
                    "sigma_l": 1.5
                },
                "firm": {
                    "alpha": 0.35,
                    "delta": 0.03
                }
            },
            "tax_parameters": {
                "baseline": {
                    "tau_c": 0.12,
                    "tau_l": 0.22
                }
            },
            "calibration_targets": {
                "cy_ratio": 0.65,
                "iy_ratio": 0.18
            }
        }
        
        filepath = self.create_test_json(test_data)
        
        try:
            params = ModelParameters.from_json(filepath)
            
            # Check that values were loaded correctly
            assert params.beta == 0.98
            assert params.sigma_c == 1.8
            assert params.sigma_l == 1.5
            assert params.alpha == 0.35
            assert params.delta == 0.03
            assert params.tau_c == 0.12
            assert params.tau_l == 0.22
            assert params.tau_l_ss == 0.22
            assert params.cy_ratio == 0.65
            assert params.iy_ratio == 0.18
            
        finally:
            os.unlink(filepath)
    
    def test_from_json_partial_override(self):
        """Test that JSON only overrides specified parameters"""
        test_data = {
            "model_parameters": {
                "household": {
                    "beta": 0.97
                }
            },
            "tax_parameters": {
                "baseline": {
                    "tau_c": 0.08
                }
            }
        }
        
        filepath = self.create_test_json(test_data)
        
        try:
            params = ModelParameters.from_json(filepath)
            
            # Check overridden values
            assert params.beta == 0.97
            assert params.tau_c == 0.08
            
            # Check that other values remain at defaults
            assert params.sigma_c == 1.5  # Default value
            assert params.alpha == 0.33   # Default value
            assert params.tau_l == 0.20   # Default value
            
        finally:
            os.unlink(filepath)
    
    def test_from_json_comment_fields_ignored(self):
        """Test that comment fields are ignored during loading"""
        test_data = {
            "model_parameters": {
                "household": {
                    "beta": 0.96,
                    "comment_beta": "This is a comment and should be ignored"
                }
            },
            "tax_parameters": {
                "baseline": {
                    "tau_c": 0.11,
                    "comment_taxes": "Another comment"
                }
            }
        }
        
        filepath = self.create_test_json(test_data)
        
        try:
            params = ModelParameters.from_json(filepath)
            
            # Check that non-comment fields were loaded
            assert params.beta == 0.96
            assert params.tau_c == 0.11
            
            # Check that comment fields don't exist as attributes
            assert not hasattr(params, 'comment_beta')
            assert not hasattr(params, 'comment_taxes')
            
        finally:
            os.unlink(filepath)
    
    def test_from_json_tau_l_consistency(self):
        """Test that tau_l and tau_l_ss are set consistently"""
        test_data = {
            "model_parameters": {},  # Empty but present
            "tax_parameters": {
                "baseline": {
                    "tau_l": 0.25
                }
            }
        }
        
        filepath = self.create_test_json(test_data)
        
        try:
            params = ModelParameters.from_json(filepath)
            
            # Both tau_l and tau_l_ss should be set to the same value
            assert params.tau_l == 0.25
            assert params.tau_l_ss == 0.25
            
        finally:
            os.unlink(filepath)
    
    def test_from_json_missing_sections(self):
        """Test that missing JSON sections don't cause errors"""
        test_data = {
            "model_parameters": {
                "household": {
                    "beta": 0.98
                }
            }
            # Missing tax_parameters and calibration_targets sections
        }
        
        filepath = self.create_test_json(test_data)
        
        try:
            params = ModelParameters.from_json(filepath)
            
            # Should load the specified parameter
            assert params.beta == 0.98
            
            # Should use defaults for missing sections
            assert params.tau_c == 0.10  # Default
            assert params.cy_ratio == 0.60  # Default
            
        finally:
            os.unlink(filepath)


class TestModelParametersEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_parameter_access(self):
        """Test that all expected parameters can be accessed"""
        params = ModelParameters()
        
        # Get all field names from the dataclass
        param_names = list(asdict(params).keys())
        
        # Check that we can access all parameters
        for param_name in param_names:
            value = getattr(params, param_name)
            assert value is not None, f"Parameter {param_name} should not be None"
    
    def test_json_nonexistent_file(self):
        """Test error handling for non-existent JSON file"""
        with pytest.raises(FileNotFoundError):
            ModelParameters.from_json("nonexistent_file.json")
    
    def test_json_invalid_format(self):
        """Test error handling for invalid JSON"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            filepath = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                ModelParameters.from_json(filepath)
        finally:
            os.unlink(filepath)
    
    def test_json_unknown_parameters(self):
        """Test that unknown parameters in JSON are ignored gracefully"""
        test_data = {
            "model_parameters": {
                "household": {
                    "beta": 0.98,
                    "unknown_parameter": 999.0  # This should be ignored
                }
            }
        }
        
        filepath = self.create_test_json(test_data)
        
        try:
            params = ModelParameters.from_json(filepath)
            
            # Should load known parameter
            assert params.beta == 0.98
            
            # Should not have unknown parameter
            assert not hasattr(params, 'unknown_parameter')
            
        finally:
            os.unlink(filepath)
    
    def create_test_json(self, content):
        """Helper to create temporary JSON file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(content, f)
            return f.name


if __name__ == "__main__":
    pytest.main([__file__])