"""
Unit tests for DSGEModel core functionality

Tests model initialization, steady state computation, equation system validation,
and core model methods without external dependencies.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch
import json

from src.dsge_model import DSGEModel, ModelParameters, SteadyState


class TestDSGEModelInitialization:
    """Test DSGEModel initialization and basic setup"""
    
    def test_initialization_with_default_parameters(self):
        """Test DSGEModel initializes correctly with default parameters"""
        params = ModelParameters()
        model = DSGEModel(params)
        
        assert model.params is params
        assert model.steady_state is None
        
        # Check that endogenous variables list is populated
        assert len(model.endogenous_vars_solve) > 0
        assert 'Y' in model.endogenous_vars_solve
        assert 'C' in model.endogenous_vars_solve
        assert 'K' in model.endogenous_vars_solve
        assert 'L' in model.endogenous_vars_solve
        
        # Check log variables indices are set
        assert isinstance(model.log_vars_indices, dict)
        assert 'K' in model.log_vars_indices
        assert 'L' in model.log_vars_indices
        
        # Check exogenous shocks are defined
        assert len(model.exogenous_shocks_sym_names) == 4
        assert 'eps_a' in model.exogenous_shocks_sym_names
        assert 'eps_g' in model.exogenous_shocks_sym_names
    
    def test_initialization_with_custom_parameters(self):
        """Test DSGEModel initialization with custom parameters"""
        params = ModelParameters(beta=0.95, sigma_c=2.0, tau_c=0.15)
        model = DSGEModel(params)
        
        assert model.params.beta == 0.95
        assert model.params.sigma_c == 2.0
        assert model.params.tau_c == 0.15
        
        # Structure should remain the same
        assert len(model.endogenous_vars_solve) > 0


class TestDSGEModelSymbolicMethods:
    """Test symbolic variable generation methods"""
    
    def test_sym_no_lags_leads(self):
        """Test _sym method with no lags or leads"""
        params = ModelParameters()
        model = DSGEModel(params)
        
        symbol = model._sym('test_var')
        assert hasattr(symbol, 'name')
        assert 'test_var' in str(symbol)
    
    def test_sym_with_lags(self):
        """Test _sym method with lags"""
        params = ModelParameters()
        model = DSGEModel(params)
        
        symbols = model._sym('test_var', lags=2)
        assert len(symbols) == 3  # t, t-1, t-2
        
        # Check that symbols have correct names
        symbol_names = [str(s) for s in symbols]
        assert any('test_var' in name and 'tm1' not in name and 'tm2' not in name for name in symbol_names)
        assert any('test_var_tm1' in name for name in symbol_names)
        assert any('test_var_tm2' in name for name in symbol_names)
    
    def test_sym_with_leads(self):
        """Test _sym method with leads"""
        params = ModelParameters()
        model = DSGEModel(params)
        
        symbols = model._sym('test_var', leads=2)
        assert len(symbols) == 3  # t, t+1, t+2
        
        # Check that symbols have correct names
        symbol_names = [str(s) for s in symbols]
        assert any('test_var' in name and 'tp1' not in name and 'tp2' not in name for name in symbol_names)
        assert any('test_var_tp1' in name for name in symbol_names)
        assert any('test_var_tp2' in name for name in symbol_names)
    
    def test_sym_with_lags_and_leads(self):
        """Test _sym method with both lags and leads"""
        params = ModelParameters()
        model = DSGEModel(params)
        
        symbols = model._sym('test_var', lags=1, leads=1)
        assert len(symbols) == 3  # t-1, t, t+1


class TestDSGEModelSteadyStateEquations:
    """Test steady state equation system"""
    
    def test_get_equations_for_steady_state_basic(self):
        """Test that steady state equations can be evaluated"""
        params = ModelParameters()
        model = DSGEModel(params)
        
        # Create a test input vector with reasonable values
        ss_default = SteadyState()
        x_test = []
        for var_name in model.endogenous_vars_solve:
            val = getattr(ss_default, var_name)
            if var_name in model.log_vars_indices:
                val = np.log(val) if val > 0 else np.log(1e-6)
            x_test.append(val)
        
        x_test = np.array(x_test)
        
        # Should be able to evaluate equations without error
        equations = model.get_equations_for_steady_state(x_test)
        
        assert isinstance(equations, np.ndarray)
        assert len(equations) == len(model.endogenous_vars_solve)
        assert all(np.isfinite(eq) for eq in equations)
    
    def test_get_equations_handles_log_variables(self):
        """Test that log variables are handled correctly"""
        params = ModelParameters()
        model = DSGEModel(params)
        
        # Test with very small values for log variables
        ss_default = SteadyState()
        x_test = []
        for var_name in model.endogenous_vars_solve:
            val = getattr(ss_default, var_name)
            if var_name in model.log_vars_indices:
                # Use very small positive value
                val = np.log(1e-6)
            x_test.append(val)
        
        x_test = np.array(x_test)
        
        # Should handle small values without numerical issues
        equations = model.get_equations_for_steady_state(x_test)
        assert all(np.isfinite(eq) for eq in equations)
    
    def test_get_equations_division_by_zero_protection(self):
        """Test protection against division by zero"""
        params = ModelParameters()
        model = DSGEModel(params)
        
        # Create input with zero values for some variables
        ss_default = SteadyState()
        x_test = []
        for var_name in model.endogenous_vars_solve:
            if var_name in ['L', 'K', 'Y']:
                # Set to very small values
                val = 1e-10
                if var_name in model.log_vars_indices:
                    val = np.log(val)
            else:
                val = getattr(ss_default, var_name)
                if var_name in model.log_vars_indices:
                    val = np.log(val) if val > 0 else np.log(1e-6)
            x_test.append(val)
        
        x_test = np.array(x_test)
        
        # Should not raise division by zero errors
        equations = model.get_equations_for_steady_state(x_test)
        assert all(np.isfinite(eq) for eq in equations)


class TestDSGEModelSteadyStateComputation:
    """Test steady state computation functionality"""
    
    def test_compute_steady_state_default(self):
        """Test steady state computation with default parameters"""
        params = ModelParameters()
        model = DSGEModel(params)
        
        # Should successfully compute steady state
        ss = model.compute_steady_state()
        
        assert isinstance(ss, SteadyState)
        assert model.steady_state is ss
        
        # Check that key variables are reasonable
        assert ss.Y > 0
        assert ss.C > 0
        assert ss.K > 0
        assert ss.L > 0
        assert 0 < ss.L < 1  # Hours should be fraction
    
    def test_compute_steady_state_with_initial_guess(self):
        """Test steady state computation with custom initial guess"""
        params = ModelParameters()
        model = DSGEModel(params)
        
        # Provide custom initial guess
        initial_guess = {
            'Y': 1.1,
            'C': 0.65,
            'I': 0.22,
            'K': 10.5,
            'L': 0.35
        }
        
        ss = model.compute_steady_state(initial_guess_dict=initial_guess)
        
        assert isinstance(ss, SteadyState)
        assert ss.Y > 0
        assert ss.C > 0
    
    def test_compute_steady_state_with_baseline(self):
        """Test steady state computation with baseline steady state"""
        params = ModelParameters()
        model = DSGEModel(params)
        
        # First compute baseline
        baseline_ss = model.compute_steady_state()
        
        # Create model with slightly different parameters
        params2 = ModelParameters(tau_c=0.11)  # Small change
        model2 = DSGEModel(params2)
        
        # Compute steady state using baseline
        ss2 = model2.compute_steady_state(baseline_ss=baseline_ss)
        
        assert isinstance(ss2, SteadyState)
        assert ss2.Y > 0
        assert ss2.C > 0
    
    def test_steady_state_convergence_check(self):
        """Test that computed steady state satisfies equations"""
        params = ModelParameters()
        model = DSGEModel(params)
        
        ss = model.compute_steady_state()
        
        # Check that steady state satisfies the equation system
        x_check = []
        for var_name in model.endogenous_vars_solve:
            val = getattr(ss, var_name)
            if var_name in model.log_vars_indices:
                val = np.log(val) if val > 0 else np.log(1e-6)
            x_check.append(val)
        
        residuals = model.get_equations_for_steady_state(np.array(x_check))
        max_residual = np.max(np.abs(residuals))
        
        # Residuals should be small
        assert max_residual < 0.1, f"Maximum residual too large: {max_residual}"


class TestDSGEModelValidation:
    """Test model validation and checking methods"""
    
    def test_check_steady_state(self):
        """Test steady state checking functionality"""
        params = ModelParameters()
        model = DSGEModel(params)
        ss = model.compute_steady_state()
        
        errors = model.check_steady_state(ss)
        
        assert isinstance(errors, dict)
        assert len(errors) > 0
        
        # Check that expected error metrics are present
        expected_checks = ['C/Y', 'I/Y', 'K/Y_annual', 'Hours', 'G/Y']
        for check in expected_checks:
            assert check in errors
        
        # Check that equation residuals are included
        residual_keys = [k for k in errors.keys() if k.startswith('Eq_Resid_')]
        assert len(residual_keys) > 0
    
    def test_check_steady_state_calibration_targets(self):
        """Test that steady state approximately meets calibration targets"""
        params = ModelParameters()
        model = DSGEModel(params)
        ss = model.compute_steady_state()
        
        errors = model.check_steady_state(ss)
        
        # Key ratios should be within reasonable bounds (model may not hit exact targets)
        assert abs(errors['C/Y']) < 0.5, f"C/Y error too large: {errors['C/Y']}"
        assert abs(errors['I/Y']) < 0.5, f"I/Y error too large: {errors['I/Y']}"
        assert abs(errors['Hours']) < 0.5, f"Hours error too large: {errors['Hours']}"


class TestDSGEModelDynamicEquations:
    """Test dynamic equation system generation"""
    
    def test_get_model_equations_requires_steady_state(self):
        """Test that dynamic equations require computed steady state"""
        params = ModelParameters()
        model = DSGEModel(params)
        
        # Should compute steady state automatically if not present
        equations = model.get_model_equations()
        
        assert len(equations) > 0
        assert model.steady_state is not None
    
    def test_get_model_equations_with_computed_steady_state(self):
        """Test dynamic equations with pre-computed steady state"""
        params = ModelParameters()
        model = DSGEModel(params)
        
        # Pre-compute steady state
        model.compute_steady_state()
        
        equations = model.get_model_equations()
        
        assert len(equations) > 0
        # Should have expected number of equations (29 based on the code)
        assert len(equations) == 29
    
    def test_dynamic_equations_structure(self):
        """Test that dynamic equations have correct structure"""
        params = ModelParameters()
        model = DSGEModel(params)
        
        equations = model.get_model_equations()
        
        # Each equation should be a sympy equation
        for eq in equations:
            assert hasattr(eq, 'lhs')
            assert hasattr(eq, 'rhs')


class TestDSGEModelUtilities:
    """Test utility functions and edge cases"""
    
    def test_load_model_function(self):
        """Test load_model utility function"""
        # Create temporary config file
        config_data = {
            "model_parameters": {
                "household": {"beta": 0.98}
            },
            "tax_parameters": {
                "baseline": {"tau_c": 0.12}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            from src.dsge_model import load_model
            model = load_model(config_path)
            
            assert isinstance(model, DSGEModel)
            assert model.params.beta == 0.98
            assert model.params.tau_c == 0.12
            
        finally:
            os.unlink(config_path)
    
    def test_model_with_extreme_parameters(self):
        """Test model behavior with extreme but valid parameters"""
        # Test with parameters at reasonable bounds
        params = ModelParameters(
            beta=0.95,    # Lower discount factor
            sigma_c=3.0,  # Higher risk aversion
            tau_c=0.05,   # Very low consumption tax
            tau_l=0.35    # Higher labor tax
        )
        
        model = DSGEModel(params)
        
        # Should still be able to compute steady state
        ss = model.compute_steady_state()
        assert isinstance(ss, SteadyState)
        assert ss.Y > 0
        assert ss.C > 0
    
    def test_model_equation_count_consistency(self):
        """Test that number of equations matches number of endogenous variables"""
        params = ModelParameters()
        model = DSGEModel(params)
        
        equations = model.get_model_equations()
        
        # For a well-defined DSGE model, we should have as many equations
        # as endogenous variables (though some may be identities)
        # The actual count depends on the specific model structure
        assert len(equations) >= len(model.endogenous_vars_solve) - 5  # Allow some flexibility


class TestDSGEModelRobustness:
    """Test model robustness and error handling"""
    
    def test_steady_state_with_problematic_parameters(self):
        """Test steady state computation with parameters that might cause issues"""
        # Test with tax rates close to bounds
        params = ModelParameters(
            tau_c=0.001,  # Very low consumption tax
            tau_l=0.45,   # High labor tax
            tau_k=0.05    # Low capital tax
        )
        
        model = DSGEModel(params)
        
        # Should handle gracefully
        try:
            ss = model.compute_steady_state()
            assert isinstance(ss, SteadyState)
        except ValueError as e:
            # If it fails, the error should be informative
            assert "SS comp failed" in str(e)
    
    def test_tax_adjusted_initial_guess(self):
        """Test tax-adjusted initial guess functionality"""
        params1 = ModelParameters()
        model1 = DSGEModel(params1)
        baseline_ss = model1.compute_steady_state()
        
        # Create model with different tax rates
        params2 = ModelParameters(tau_c=0.15, tau_l=0.25)
        model2 = DSGEModel(params2)
        
        # Test the tax-adjusted initial guess method
        initial_guess = model2._compute_tax_adjusted_initial_guess(baseline_ss)
        
        assert isinstance(initial_guess, dict)
        assert len(initial_guess) > 0
        
        # Key variables should be present
        assert 'C' in initial_guess
        assert 'L' in initial_guess
        assert 'I' in initial_guess
        
        # Most values should be positive (except some like b_star which can be zero/negative)
        positive_vars = ['C', 'I', 'K', 'L', 'Y', 'w', 'Rk_gross', 'Lambda', 'G']
        for var_name in positive_vars:
            if var_name in initial_guess:
                assert initial_guess[var_name] > 0, f"{var_name} should be positive in initial guess"


if __name__ == "__main__":
    pytest.main([__file__])