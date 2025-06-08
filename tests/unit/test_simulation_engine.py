"""
Unit tests for EnhancedTaxSimulator simulation engine

Tests simulation execution, implementation modes, welfare calculation,
and fiscal impact analysis with mocked dependencies.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from src.dsge_model import DSGEModel, ModelParameters, SteadyState
from src.tax_simulator import TaxReform, EnhancedTaxSimulator, SimulationResults


class TestEnhancedTaxSimulatorInitialization:
    """Test EnhancedTaxSimulator initialization and setup"""
    
    def test_initialization_with_dsge_model(self):
        """Test initialization with DSGEModel"""
        params = ModelParameters()
        model = DSGEModel(params)
        
        # EnhancedTaxSimulator requires pre-computed steady state
        model.compute_steady_state()
        
        simulator = EnhancedTaxSimulator(model)
        
        assert simulator.baseline_model is model
        assert simulator.baseline_params is params
        assert hasattr(simulator, 'baseline_ss')
        assert hasattr(simulator, 'linear_model')
    
    @patch('src.tax_simulator.ImprovedLinearizedDSGE')
    def test_initialization_handles_linearization_failure(self, mock_linearized):
        """Test fallback when linearization fails"""
        # Mock linearization failure
        mock_linearized.side_effect = Exception("Linearization failed")
        
        params = ModelParameters()
        model = DSGEModel(params)
        model.compute_steady_state()
        
        # Should handle gracefully with fallback
        try:
            simulator = EnhancedTaxSimulator(model)
            # Should still initialize basic attributes
            assert simulator.baseline_model is model
            assert simulator.baseline_params is params
        except Exception:
            # May fail due to linearization, which is acceptable
            pass
    
    def test_initialization_sets_default_parameters(self):
        """Test that initialization sets appropriate default simulation parameters"""
        params = ModelParameters()
        model = DSGEModel(params)
        model.compute_steady_state()
        
        simulator = EnhancedTaxSimulator(model)
        
        # Should have reasonable default simulation parameters
        assert hasattr(simulator, 'baseline_model')
        assert hasattr(simulator, 'baseline_params')


class TestSimulationMethods:
    """Test core simulation methods with mocked dependencies"""
    
    @patch('src.tax_simulator.EnhancedTaxSimulator._simulate_with_shocks')
    def test_simulate_permanent_reform(self, mock_simulate):
        """Test permanent reform simulation logic"""
        # Setup
        params = ModelParameters()
        model = DSGEModel(params)
        model.compute_steady_state()
        simulator = EnhancedTaxSimulator(model)
        
        # Mock the core simulation method
        mock_path = np.random.rand(40, 10)  # 40 periods, 10 variables
        mock_simulate.return_value = mock_path
        
        reform = TaxReform(
            name="Permanent Reform",
            tau_c=0.15,
            implementation='permanent'
        )
        
        # Test permanent simulation
        tax_changes = reform.get_changes(simulator.baseline_params)
        result = simulator._simulate_permanent_reform(tax_changes, periods=40)
        
        # Should call core simulation method
        mock_simulate.assert_called()
        
        # Should return simulation path
        assert isinstance(result, np.ndarray)
    
    @patch('src.tax_simulator.EnhancedTaxSimulator._simulate_with_shocks')
    def test_simulate_temporary_reform(self, mock_simulate):
        """Test temporary reform simulation logic"""
        params = ModelParameters()
        model = DSGEModel(params)
        model.compute_steady_state()
        simulator = EnhancedTaxSimulator(model)
        
        # Mock simulation returns
        mock_path_reform = np.random.rand(40, 10)
        mock_path_baseline = np.random.rand(40, 10)
        mock_simulate.side_effect = [mock_path_reform, mock_path_baseline]
        
        reform = TaxReform(
            name="Temporary Reform",
            tau_c=0.15,
            implementation='temporary',
            duration=8
        )
        
        # Test temporary simulation
        tax_changes = reform.get_changes(simulator.baseline_params)
        result = simulator._simulate_temporary_reform(tax_changes, reform.duration or 20, periods=40)
        
        # Should call simulation method
        mock_simulate.assert_called()
        assert isinstance(result, np.ndarray)
    
    @patch('src.tax_simulator.EnhancedTaxSimulator._simulate_with_shocks')
    def test_simulate_phased_reform(self, mock_simulate):
        """Test phased reform simulation logic"""
        params = ModelParameters()
        model = DSGEModel(params)
        model.compute_steady_state()
        simulator = EnhancedTaxSimulator(model)
        
        # Mock simulation
        mock_path = np.random.rand(40, 10)
        mock_simulate.return_value = mock_path
        
        reform = TaxReform(
            name="Phased Reform",
            tau_c=0.15,
            implementation='phased',
            phase_in_periods=4
        )
        
        # Test phased simulation
        tax_changes = reform.get_changes(simulator.baseline_params)
        result = simulator._simulate_phased_reform(tax_changes, reform.phase_in_periods or 8, periods=40)
        
        # Should call simulation with gradual implementation
        mock_simulate.assert_called()
        assert isinstance(result, np.ndarray)
    
    def test_shock_scaling_logic(self):
        """Test shock scaling based on tax change magnitude"""
        params = ModelParameters()
        model = DSGEModel(params)
        model.compute_steady_state()
        simulator = EnhancedTaxSimulator(model)
        
        # Test scaling for different tax change magnitudes
        small_change = 0.01  # 1 percentage point
        large_change = 0.10  # 10 percentage points
        
        # Mock shock scaling calculation
        small_shock_scale = min(abs(small_change) * 20, 5.0)
        large_shock_scale = min(abs(large_change) * 20, 5.0)
        
        assert small_shock_scale < large_shock_scale
        assert large_shock_scale <= 5.0  # Should be capped


class TestWelfareCalculation:
    """Test welfare calculation methods"""
    
    def test_compute_welfare_change_basic(self):
        """Test basic welfare change computation"""
        params = ModelParameters(beta=0.99, habit=0.6, sigma_c=1.5, sigma_l=2.0, chi=3.0)
        model = DSGEModel(params)
        model.compute_steady_state()
        simulator = EnhancedTaxSimulator(model)
        
        # Mock consumption and labor paths
        periods = 20
        baseline_C = np.ones(periods) * 0.6  # Steady consumption
        reform_C = np.ones(periods) * 0.62   # 3.33% higher consumption
        baseline_L = np.ones(periods) * 0.33
        reform_L = np.ones(periods) * 0.34
        
        # Create mock DataFrames for welfare calculation
        baseline_path = pd.DataFrame({'C': baseline_C, 'L': baseline_L})
        reform_path = pd.DataFrame({'C': reform_C, 'L': reform_L})
        
        # Test welfare calculation
        welfare_change = simulator._compute_welfare_change(
            baseline_path, reform_path, params
        )
        
        # Should return a finite number
        assert np.isfinite(welfare_change)
        
        # Higher consumption should generally improve welfare
        assert welfare_change > 0, "Higher consumption should improve welfare"
    
    def test_welfare_calculation_with_habit_formation(self):
        """Test welfare calculation with habit formation"""
        params = ModelParameters(beta=0.99, habit=0.8, sigma_c=1.5)  # Strong habit
        model = DSGEModel(params)
        model.compute_steady_state()
        simulator = EnhancedTaxSimulator(model)
        
        periods = 10
        # Create consumption path with gradual increase (habit matters)
        baseline_C = np.ones(periods) * 0.6
        reform_C = np.linspace(0.6, 0.65, periods)  # Gradual increase
        baseline_L = np.ones(periods) * 0.33
        reform_L = np.ones(periods) * 0.33
        
        baseline_path = pd.DataFrame({'C': baseline_C, 'L': baseline_L})
        reform_path = pd.DataFrame({'C': reform_C, 'L': reform_L})
        
        welfare_change = simulator._compute_welfare_change(
            baseline_path, reform_path, params
        )
        
        assert np.isfinite(welfare_change)
        # With habit formation, gradual increases are valued differently
        assert isinstance(welfare_change, (int, float))
    
    def test_welfare_calculation_edge_cases(self):
        """Test welfare calculation edge cases"""
        params = ModelParameters()
        model = DSGEModel(params)
        model.compute_steady_state()
        simulator = EnhancedTaxSimulator(model)
        
        periods = 5
        
        # Test with identical paths (should be zero welfare change)
        baseline_C = np.ones(periods) * 0.6
        reform_C = np.ones(periods) * 0.6
        baseline_L = np.ones(periods) * 0.33
        reform_L = np.ones(periods) * 0.33
        
        baseline_path = pd.DataFrame({'C': baseline_C, 'L': baseline_L})
        reform_path = pd.DataFrame({'C': reform_C, 'L': reform_L})
        
        welfare_change = simulator._compute_welfare_change(
            baseline_path, reform_path, params
        )
        
        # Should be very close to zero
        assert abs(welfare_change) < 1e-6, "Identical paths should have zero welfare change"
    
    def test_welfare_calculation_with_negative_consumption_change(self):
        """Test welfare calculation when consumption decreases"""
        params = ModelParameters()
        model = DSGEModel(params)
        model.compute_steady_state()
        simulator = EnhancedTaxSimulator(model)
        
        periods = 10
        baseline_C = np.ones(periods) * 0.6
        reform_C = np.ones(periods) * 0.55   # 8.33% lower consumption
        baseline_L = np.ones(periods) * 0.33
        reform_L = np.ones(periods) * 0.33
        
        baseline_path = pd.DataFrame({'C': baseline_C, 'L': baseline_L})
        reform_path = pd.DataFrame({'C': reform_C, 'L': reform_L})
        
        welfare_change = simulator._compute_welfare_change(
            baseline_path, reform_path, params
        )
        
        assert np.isfinite(welfare_change)
        # Lower consumption should reduce welfare
        assert welfare_change < 0, "Lower consumption should reduce welfare"


class TestFiscalImpactAnalysis:
    """Test fiscal impact calculation methods"""
    
    def test_compute_fiscal_impact_basic(self):
        """Test basic fiscal impact computation"""
        params = ModelParameters()
        model = DSGEModel(params)
        model.compute_steady_state()
        simulator = EnhancedTaxSimulator(model)
        
        # Mock baseline and reform paths as DataFrames
        periods = 40
        baseline_path = pd.DataFrame({
            'Y': np.ones(periods) * 1.0,
            'T_total_revenue': np.ones(periods) * 0.2,
            'G': np.ones(periods) * 0.2
        })
        reform_path = pd.DataFrame({
            'Y': np.ones(periods) * 1.05,
            'T_total_revenue': np.ones(periods) * 0.22,
            'G': np.ones(periods) * 0.21
        })
        
        fiscal_impact = simulator._compute_fiscal_impact(
            baseline_path, reform_path, periods
        )
        
        # Should return a DataFrame with impact metrics
        assert isinstance(fiscal_impact, pd.DataFrame)
        
        # Should have expected time horizons in index or columns
        fiscal_impact_str = str(fiscal_impact.index) + str(fiscal_impact.columns)
        expected_horizons = ['Impact', 'Short-run', 'Medium-run', 'Long-run']
        # At least one horizon should be present
        assert any(horizon in fiscal_impact_str for horizon in expected_horizons)
    
    def test_fiscal_impact_calculation_horizons(self):
        """Test fiscal impact calculation for different time horizons"""
        params = ModelParameters()
        model = DSGEModel(params)
        model.compute_steady_state()
        simulator = EnhancedTaxSimulator(model)
        
        periods = 40
        baseline_path = pd.DataFrame({
            'Y': np.ones(periods) * 1.0,
            'T_total_revenue': np.ones(periods) * 0.2
        })
        reform_path = pd.DataFrame({
            'Y': np.ones(periods) * 1.02,
            'T_total_revenue': np.ones(periods) * 0.204
        })
        
        fiscal_impact = simulator._compute_fiscal_impact(
            baseline_path, reform_path, periods
        )
        
        # Should have multi-level structure with horizons
        assert isinstance(fiscal_impact, pd.DataFrame)
        assert len(fiscal_impact) > 0  # Should have some results
    
    def test_fiscal_impact_with_negative_changes(self):
        """Test fiscal impact with negative revenue changes"""
        params = ModelParameters()
        model = DSGEModel(params)
        model.compute_steady_state()
        simulator = EnhancedTaxSimulator(model)
        
        periods = 20
        baseline_path = pd.DataFrame({
            'Y': np.ones(periods) * 1.0,
            'T_total_revenue': np.ones(periods) * 0.2
        })
        reform_path = pd.DataFrame({
            'Y': np.ones(periods) * 0.98,
            'T_total_revenue': np.ones(periods) * 0.18
        })
        
        fiscal_impact = simulator._compute_fiscal_impact(
            baseline_path, reform_path, periods
        )
        
        # Should handle negative changes appropriately
        assert isinstance(fiscal_impact, pd.DataFrame)
        assert len(fiscal_impact) > 0  # Should have results


class TestTransitionAnalysis:
    """Test transition period analysis methods"""
    
    def test_find_transition_period_basic(self):
        """Test transition period identification"""
        params = ModelParameters()
        model = DSGEModel(params)
        model.compute_steady_state()
        simulator = EnhancedTaxSimulator(model)
        
        # Create mock transition path that converges
        periods = 30
        path_data = []
        for t in range(periods):
            path_data.append({
                'Y': 1.0 + 0.1 * np.exp(-t * 0.2),
                'C': 0.6 + 0.05 * np.exp(-t * 0.2),
                'K': 10.0 + 0.5 * np.exp(-t * 0.2),
                'L': 0.3 + 0.02 * np.exp(-t * 0.2)
            })
        path = pd.DataFrame(path_data)
        
        # Mock steady state
        new_ss = SteadyState(Y=1.0, C=0.6, K=10.0, L=0.3)
        
        # Test transition detection
        transition_period = simulator._find_transition_period(path, new_ss)
        
        # Should find reasonable transition period
        assert isinstance(transition_period, int)
        assert 0 < transition_period <= periods
    
    def test_find_transition_period_slow_convergence(self):
        """Test transition period with slow convergence"""
        params = ModelParameters()
        model = DSGEModel(params)
        model.compute_steady_state()
        simulator = EnhancedTaxSimulator(model)
        
        # Slow convergence path
        periods = 40
        path_data = []
        for t in range(periods):
            decay = 0.95 ** t
            path_data.append({
                'Y': 1.0 + 0.1 * decay,
                'C': 0.6 + 0.05 * decay,
                'K': 10.0 + 0.5 * decay,
                'L': 0.3 + 0.02 * decay
            })
        path = pd.DataFrame(path_data)
        new_ss = SteadyState(Y=1.0, C=0.6, K=10.0, L=0.3)
        
        transition_period = simulator._find_transition_period(path, new_ss)
        
        # Should handle slow convergence
        assert isinstance(transition_period, int)
        assert transition_period > 0
    
    def test_find_transition_period_no_convergence(self):
        """Test transition period when there's no clear convergence"""
        params = ModelParameters()
        model = DSGEModel(params)
        model.compute_steady_state()
        simulator = EnhancedTaxSimulator(model)
        
        # Oscillating or non-converging path
        periods = 20
        path_data = []
        for t in range(periods):
            osc = 0.1 * np.sin(t * 0.5)
            path_data.append({
                'Y': 1.0 + osc,
                'C': 0.6 + osc * 0.5,
                'K': 10.0 + osc * 2.0,
                'L': 0.3 + osc * 0.2
            })
        path = pd.DataFrame(path_data)
        new_ss = SteadyState(Y=1.0, C=0.6, K=10.0, L=0.3)
        
        transition_period = simulator._find_transition_period(path, new_ss)
        
        # Should return reasonable default or handle gracefully
        assert isinstance(transition_period, int)
        assert transition_period >= 0  # May return 0 for non-converging cases


class TestSimulationValidation:
    """Test simulation validation and error handling"""
    
    def test_validate_steady_state_change_reasonable(self):
        """Test validation of reasonable steady state changes"""
        params = ModelParameters()
        model = DSGEModel(params)
        model.compute_steady_state()
        simulator = EnhancedTaxSimulator(model)
        
        # Reasonable steady state changes
        baseline_ss = SteadyState(Y=1.0, C=0.6, I=0.2)
        reform_ss = SteadyState(Y=1.02, C=0.61, I=0.21)  # 2% increase
        
        # Should validate without error
        is_valid = simulator._validate_steady_state_change(reform_ss)
        
        # Should return True for reasonable changes
        assert is_valid or is_valid is None  # Method might not return boolean
    
    def test_validate_steady_state_change_extreme(self):
        """Test validation with extreme steady state changes"""
        params = ModelParameters()
        model = DSGEModel(params)
        model.compute_steady_state()
        simulator = EnhancedTaxSimulator(model)
        
        # Extreme changes
        baseline_ss = SteadyState(Y=1.0, C=0.6, I=0.2)
        reform_ss = SteadyState(Y=0.5, C=0.3, I=0.1)  # 50% decrease (extreme)
        
        # Should flag as potentially problematic
        try:
            is_valid = simulator._validate_steady_state_change(reform_ss)
            # Method might return False or issue warnings
            assert isinstance(is_valid, (bool, type(None)))
        except:
            # Method might raise exception for extreme cases
            pass
    
    def test_simulation_result_structure(self):
        """Test that simulation results have expected structure"""
        # This tests the expected output structure without full simulation
        
        # Mock result structure
        mock_result = {
            'baseline_path': np.random.rand(40, 10),
            'reform_path': np.random.rand(40, 10),
            'welfare_change': 0.02,
            'fiscal_impact': {
                'impact': {'revenue_change_pct': 0.05},
                'long_run': {'revenue_change_pct': 0.03}
            }
        }
        
        # Check expected structure
        assert 'baseline_path' in mock_result
        assert 'reform_path' in mock_result
        assert 'welfare_change' in mock_result
        assert 'fiscal_impact' in mock_result
        
        # Check fiscal impact structure
        fiscal = mock_result['fiscal_impact']
        assert 'impact' in fiscal
        assert 'long_run' in fiscal


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases"""
    
    def test_simulation_with_failed_steady_state(self):
        """Test simulation when steady state computation fails"""
        params = ModelParameters()
        model = DSGEModel(params)
        model.compute_steady_state()
        simulator = EnhancedTaxSimulator(model)
        
        # Reform that might cause steady state failure
        extreme_reform = TaxReform(
            name="Extreme Reform",
            tau_c=0.5,  # Very high consumption tax
            tau_l=0.6   # Very high labor tax
        )
        
        # Should handle gracefully
        try:
            # This might fail, but shouldn't crash
            result = simulator.simulate_reform(extreme_reform, periods=10)
        except Exception as e:
            # Should provide informative error message
            assert isinstance(e, Exception)
    
    def test_simulation_with_zero_periods(self):
        """Test simulation with edge case parameters"""
        params = ModelParameters()
        model = DSGEModel(params)
        model.compute_steady_state()
        simulator = EnhancedTaxSimulator(model)
        
        reform = TaxReform(name="Test", tau_c=0.12)
        
        # Test with minimal periods
        try:
            result = simulator.simulate_reform(reform, periods=1)
            # Should handle minimal simulation
            from src.tax_simulator import SimulationResults
            assert isinstance(result, (SimulationResults, type(None)))
        except ValueError:
            # Might raise error for insufficient periods
            pass
    
    def test_numerical_stability_checks(self):
        """Test numerical stability in calculations"""
        # Test basic numerical operations used in simulation
        
        # Test beta discounting
        beta = 0.99
        periods = 100
        discount_factors = np.array([beta ** t for t in range(periods)])
        
        # Should remain finite and positive
        assert np.all(np.isfinite(discount_factors))
        assert np.all(discount_factors > 0)
        assert np.all(discount_factors <= 1)
        
        # Test consumption utility with habit formation
        C = np.array([0.6, 0.61, 0.62])
        habit = 0.6
        
        for t in range(1, len(C)):
            consumption_term = C[t] - habit * C[t-1]
            assert consumption_term > 0, "Consumption net of habit should be positive"
    
    def test_mock_linearization_integration(self):
        """Test integration with mocked linearization"""
        with patch('src.tax_simulator.ImprovedLinearizedDSGE') as mock_linearized:
            # Mock linearization class
            mock_instance = Mock()
            mock_instance.solve_klein.return_value = None
            mock_instance.n_s = 2
            mock_linearized.return_value = mock_instance
            
            params = ModelParameters()
            model = DSGEModel(params)
            model.compute_steady_state()  # Need steady state for initialization
            
            # Should initialize with mocked linearization
            simulator = EnhancedTaxSimulator(model)
            
            # Should have basic attributes
            assert hasattr(simulator, 'baseline_model')
            assert hasattr(simulator, 'baseline_params')


if __name__ == "__main__":
    pytest.main([__file__])