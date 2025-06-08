"""
Unit tests for SimulationResults and analysis functionality

Tests result containers, aggregate effect computation, comparison methods,
and pandas DataFrame integration with correct SimulationResults structure.
"""

import pytest
import numpy as np
import pandas as pd
from dataclasses import asdict

from src.dsge_model import SteadyState
from src.tax_simulator import SimulationResults


class TestSimulationResultsBasic:
    """Test basic SimulationResults functionality"""
    
    def test_default_initialization(self):
        """Test that SimulationResults initializes with required fields"""
        # Create mock data as pandas DataFrames
        var_names = ['Y', 'C', 'I', 'L']
        baseline_path = pd.DataFrame(np.random.rand(40, 4), columns=var_names)
        reform_path = pd.DataFrame(np.random.rand(40, 4), columns=var_names)
        baseline_ss = SteadyState()
        reform_ss = SteadyState()
        
        # Create mock fiscal impact DataFrame
        fiscal_impact = pd.DataFrame({
            'horizon': ['impact', 'short_run', 'long_run'],
            'revenue_change_pct': [0.05, 0.03, 0.02]
        })
        
        results = SimulationResults(
            name="Test Reform",
            baseline_path=baseline_path,
            reform_path=reform_path,
            steady_state_baseline=baseline_ss,
            steady_state_reform=reform_ss,
            welfare_change=0.02,
            fiscal_impact=fiscal_impact,
            transition_periods=10
        )
        
        assert results.baseline_path.equals(baseline_path)
        assert results.reform_path.equals(reform_path)
        assert results.steady_state_baseline is baseline_ss
        assert results.steady_state_reform is reform_ss
        assert results.welfare_change == 0.02
        assert results.transition_periods == 10
    
    def test_path_dimensions_consistency(self):
        """Test that baseline and reform paths have consistent dimensions"""
        var_names = ['Y', 'C', 'I']
        periods = 30
        
        baseline_path = pd.DataFrame(np.random.rand(periods, 3), columns=var_names)
        reform_path = pd.DataFrame(np.random.rand(periods, 3), columns=var_names)
        
        fiscal_impact = pd.DataFrame({'horizon': ['impact'], 'revenue_change_pct': [0.05]})
        
        results = SimulationResults(
            name="Test",
            baseline_path=baseline_path,
            reform_path=reform_path,
            steady_state_baseline=SteadyState(),
            steady_state_reform=SteadyState(),
            welfare_change=0.01,
            fiscal_impact=fiscal_impact,
            transition_periods=5
        )
        
        # Paths should have same dimensions
        assert results.baseline_path.shape == results.reform_path.shape
        assert len(results.baseline_path) == periods
        assert len(results.baseline_path.columns) == 3


class TestSimulationResultsComputeAggregateEffects:
    """Test aggregate effects computation"""
    
    def test_compute_aggregate_effects_basic(self):
        """Test basic aggregate effects computation"""
        var_names = ['Y', 'C']
        periods = 20
        
        # Create paths with known differences
        baseline_data = np.ones((periods, 2))
        reform_data = np.ones((periods, 2)) * 1.05  # 5% increase
        
        baseline_path = pd.DataFrame(baseline_data, columns=var_names)
        reform_path = pd.DataFrame(reform_data, columns=var_names)
        
        fiscal_impact = pd.DataFrame({'horizon': ['impact'], 'revenue_change_pct': [0.05]})
        
        results = SimulationResults(
            name="Test Reform",
            baseline_path=baseline_path,
            reform_path=reform_path,
            steady_state_baseline=SteadyState(),
            steady_state_reform=SteadyState(),
            welfare_change=0.02,
            fiscal_impact=fiscal_impact,
            transition_periods=5
        )
        
        # Test aggregate effects computation
        aggregate_effects = results.compute_aggregate_effects(variables=['Y', 'C'])
        
        # Should return DataFrame with effects
        assert isinstance(aggregate_effects, pd.DataFrame)
        
        # Should include expected variables
        assert 'Y' in aggregate_effects.index
        assert 'C' in aggregate_effects.index
        
        # Should include expected columns
        expected_cols = ['Baseline', 'Reform', 'Change', '% Change']
        for col in expected_cols:
            assert col in aggregate_effects.columns
    
    def test_compute_aggregate_effects_with_time_windows(self):
        """Test aggregate effects over different time windows"""
        var_names = ['Y', 'C']
        periods = 40
        
        baseline_path = pd.DataFrame(np.ones((periods, 2)), columns=var_names)
        reform_path = pd.DataFrame(np.ones((periods, 2)) * 1.03, columns=var_names)
        
        fiscal_impact = pd.DataFrame({'horizon': ['impact'], 'revenue_change_pct': [0.03]})
        
        results = SimulationResults(
            name="Test",
            baseline_path=baseline_path,
            reform_path=reform_path,
            steady_state_baseline=SteadyState(),
            steady_state_reform=SteadyState(),
            welfare_change=0.015,
            fiscal_impact=fiscal_impact,
            transition_periods=8
        )
        
        # Test effects over different windows
        effects_10 = results.compute_aggregate_effects(variables=['Y'], periods=10)
        effects_20 = results.compute_aggregate_effects(variables=['Y'], periods=20)
        
        # Should return DataFrames
        assert isinstance(effects_10, pd.DataFrame)
        assert isinstance(effects_20, pd.DataFrame)
        
        # Should include Y variable
        assert 'Y' in effects_10.index
        assert 'Y' in effects_20.index
    
    def test_compute_aggregate_effects_percentage_calculations(self):
        """Test percentage deviation calculations"""
        var_names = ['Y']
        periods = 15
        
        baseline_data = np.array([[1.0]] * periods)
        reform_data = np.array([[1.1]] * periods)  # Exactly 10% increase
        
        baseline_path = pd.DataFrame(baseline_data, columns=var_names)
        reform_path = pd.DataFrame(reform_data, columns=var_names)
        
        fiscal_impact = pd.DataFrame({'horizon': ['impact'], 'revenue_change_pct': [0.1]})
        
        results = SimulationResults(
            name="10% Reform",
            baseline_path=baseline_path,
            reform_path=reform_path,
            steady_state_baseline=SteadyState(),
            steady_state_reform=SteadyState(),
            welfare_change=0.05,
            fiscal_impact=fiscal_impact,
            transition_periods=3
        )
        
        effects = results.compute_aggregate_effects(variables=['Y'])
        
        # Check percentage calculation
        y_pct_change = effects.loc['Y', '% Change']
        assert abs(y_pct_change - 10.0) < 1e-10, f"Expected 10%, got {y_pct_change}"


class TestSimulationResultsDataAccess:
    """Test data access and manipulation methods"""
    
    def test_dataframe_attribute_access(self):
        """Test pandas DataFrame attribute access"""
        var_names = ['Y', 'C', 'I']
        baseline_path = pd.DataFrame(np.random.rand(10, 3), columns=var_names)
        reform_path = pd.DataFrame(np.random.rand(10, 3), columns=var_names)
        
        fiscal_impact = pd.DataFrame({
            'horizon': ['impact', 'long_run'],
            'revenue_change_pct': [0.05, 0.03]
        })
        
        results = SimulationResults(
            name="Access Test",
            baseline_path=baseline_path,
            reform_path=reform_path,
            steady_state_baseline=SteadyState(),
            steady_state_reform=SteadyState(),
            welfare_change=0.02,
            fiscal_impact=fiscal_impact,
            transition_periods=5
        )
        
        # Test DataFrame properties
        assert isinstance(results.baseline_path, pd.DataFrame)
        assert isinstance(results.reform_path, pd.DataFrame)
        assert isinstance(results.fiscal_impact, pd.DataFrame)
        
        # Test column access
        assert 'Y' in results.baseline_path.columns
        assert 'C' in results.reform_path.columns
        
        # Test data access
        y_baseline_first = results.baseline_path['Y'].iloc[0]
        assert isinstance(y_baseline_first, (int, float, np.number))
    
    def test_path_slicing_and_indexing(self):
        """Test slicing and indexing of simulation paths"""
        var_names = ['Y', 'C']
        periods = 40
        
        baseline_path = pd.DataFrame(np.random.rand(periods, 2), columns=var_names)
        reform_path = pd.DataFrame(np.random.rand(periods, 2), columns=var_names)
        
        fiscal_impact = pd.DataFrame({'horizon': ['impact'], 'revenue_change_pct': [0.02]})
        
        results = SimulationResults(
            name="Slicing Test",
            baseline_path=baseline_path,
            reform_path=reform_path,
            steady_state_baseline=SteadyState(),
            steady_state_reform=SteadyState(),
            welfare_change=0.01,
            fiscal_impact=fiscal_impact,
            transition_periods=10
        )
        
        # Test time slicing
        first_year = results.baseline_path.iloc[:4]  # First 4 quarters
        assert len(first_year) == 4
        assert len(first_year.columns) == 2
        
        # Test variable selection
        y_series_baseline = results.baseline_path['Y']
        y_series_reform = results.reform_path['Y']
        
        assert len(y_series_baseline) == periods
        assert len(y_series_reform) == periods
        
        # Test difference calculation
        y_difference = y_series_reform - y_series_baseline
        assert len(y_difference) == periods
    
    def test_dataclass_conversion(self):
        """Test conversion to dictionary (dataclass functionality)"""
        var_names = ['Y']
        baseline_path = pd.DataFrame([[1.0]], columns=var_names)
        reform_path = pd.DataFrame([[1.05]], columns=var_names)
        fiscal_impact = pd.DataFrame({'horizon': ['impact'], 'revenue_change_pct': [0.05]})
        
        results = SimulationResults(
            name="Dict Test",
            baseline_path=baseline_path,
            reform_path=reform_path,
            steady_state_baseline=SteadyState(),
            steady_state_reform=SteadyState(),
            welfare_change=0.025,
            fiscal_impact=fiscal_impact,
            transition_periods=1
        )
        
        # Test dataclass functionality
        results_dict = asdict(results)
        
        # Should include all fields
        expected_fields = ['name', 'baseline_path', 'reform_path', 'steady_state_baseline', 
                          'steady_state_reform', 'welfare_change', 'fiscal_impact', 'transition_periods']
        
        for field in expected_fields:
            assert field in results_dict, f"Field {field} should be accessible"


class TestSimulationResultsAnalysisMethods:
    """Test analysis and comparison methods"""
    
    def test_steady_state_comparison(self):
        """Test steady state comparison functionality"""
        baseline_ss = SteadyState(Y=1.0, C=0.6, I=0.2)
        reform_ss = SteadyState(Y=1.03, C=0.62, I=0.21)
        
        var_names = ['Y']
        baseline_path = pd.DataFrame([[1.0]], columns=var_names)
        reform_path = pd.DataFrame([[1.03]], columns=var_names)
        fiscal_impact = pd.DataFrame({'horizon': ['impact'], 'revenue_change_pct': [0.03]})
        
        results = SimulationResults(
            name="SS Comparison",
            baseline_path=baseline_path,
            reform_path=reform_path,
            steady_state_baseline=baseline_ss,
            steady_state_reform=reform_ss,
            welfare_change=0.015,
            fiscal_impact=fiscal_impact,
            transition_periods=1
        )
        
        # Test steady state changes
        y_change = results.steady_state_reform.Y - results.steady_state_baseline.Y
        c_change = results.steady_state_reform.C - results.steady_state_baseline.C
        
        assert y_change > 0, "Output should increase"
        assert c_change > 0, "Consumption should increase"
        
        # Test percentage changes
        y_pct_change = (results.steady_state_reform.Y / results.steady_state_baseline.Y - 1) * 100
        assert 2.5 < y_pct_change < 3.5, "Output change should be around 3%"
    
    def test_welfare_and_fiscal_interpretation(self):
        """Test welfare and fiscal impact interpretation"""
        var_names = ['Y']
        baseline_path = pd.DataFrame([[1.0]], columns=var_names)
        reform_path = pd.DataFrame([[1.02]], columns=var_names)
        
        fiscal_impact = pd.DataFrame({
            'horizon': ['impact', 'short_run', 'long_run'],
            'revenue_change_pct': [0.08, 0.06, 0.04],
            'debt_change_pct': [-0.02, -0.01, 0.01]
        })
        
        results = SimulationResults(
            name="Welfare Test",
            baseline_path=baseline_path,
            reform_path=reform_path,
            steady_state_baseline=SteadyState(),
            steady_state_reform=SteadyState(),
            welfare_change=0.025,  # 2.5% consumption equivalent
            fiscal_impact=fiscal_impact,
            transition_periods=1
        )
        
        # Test welfare interpretation
        welfare_pct = results.welfare_change * 100
        assert welfare_pct == 2.5, "Welfare change should be 2.5%"
        
        # Test fiscal impact access
        assert 'revenue_change_pct' in results.fiscal_impact.columns
        assert 'debt_change_pct' in results.fiscal_impact.columns
        
        # Test impact vs long-run comparison
        impact_revenue = results.fiscal_impact[results.fiscal_impact['horizon'] == 'impact']['revenue_change_pct'].iloc[0]
        longrun_revenue = results.fiscal_impact[results.fiscal_impact['horizon'] == 'long_run']['revenue_change_pct'].iloc[0]
        
        assert impact_revenue > longrun_revenue, "Revenue effect should decline over time"


class TestSimulationResultsComparison:
    """Test methods for comparing different simulation results"""
    
    def test_results_comparison_structure(self):
        """Test structure for comparing multiple simulation results"""
        var_names = ['Y']
        baseline_path = pd.DataFrame([[1.0]], columns=var_names)
        fiscal_impact_template = pd.DataFrame({'horizon': ['impact'], 'revenue_change_pct': [0.0]})
        
        # Reform 1: Moderate changes
        reform1_path = pd.DataFrame([[1.02]], columns=var_names)
        results1 = SimulationResults(
            name="Moderate Reform",
            baseline_path=baseline_path,
            reform_path=reform1_path,
            steady_state_baseline=SteadyState(Y=1.0),
            steady_state_reform=SteadyState(Y=1.02),
            welfare_change=0.01,
            fiscal_impact=fiscal_impact_template.copy(),
            transition_periods=1
        )
        
        # Reform 2: Larger changes
        reform2_path = pd.DataFrame([[1.05]], columns=var_names)
        results2 = SimulationResults(
            name="Large Reform",
            baseline_path=baseline_path,
            reform_path=reform2_path,
            steady_state_baseline=SteadyState(Y=1.0),
            steady_state_reform=SteadyState(Y=1.05),
            welfare_change=0.025,
            fiscal_impact=fiscal_impact_template.copy(),
            transition_periods=1
        )
        
        # Test comparison logic
        assert results2.welfare_change > results1.welfare_change
        assert results2.steady_state_reform.Y > results1.steady_state_reform.Y
        assert results2.name != results1.name
    
    def test_results_ranking_by_welfare(self):
        """Test ranking of results by welfare criteria"""
        var_names = ['Y']
        baseline_path = pd.DataFrame([[1.0]], columns=var_names)
        reform_path = pd.DataFrame([[1.01]], columns=var_names)
        fiscal_impact = pd.DataFrame({'horizon': ['impact'], 'revenue_change_pct': [0.01]})
        
        # Create multiple results with different welfare outcomes
        welfare_values = [0.01, -0.005, 0.03, 0.015]
        results_list = []
        
        for i, welfare in enumerate(welfare_values):
            results = SimulationResults(
                name=f"Reform {i+1}",
                baseline_path=baseline_path,
                reform_path=reform_path,
                steady_state_baseline=SteadyState(),
                steady_state_reform=SteadyState(),
                welfare_change=welfare,
                fiscal_impact=fiscal_impact.copy(),
                transition_periods=1
            )
            results_list.append(results)
        
        # Sort by welfare
        sorted_results = sorted(results_list, key=lambda x: x.welfare_change, reverse=True)
        
        # Check sorting
        assert sorted_results[0].welfare_change == 0.03  # Best
        assert sorted_results[-1].welfare_change == -0.005  # Worst
        
        # Check that all results are present
        assert len(sorted_results) == 4


class TestSimulationResultsEdgeCases:
    """Test edge cases and error handling"""
    
    def test_minimal_data_handling(self):
        """Test handling of minimal simulation data"""
        var_names = ['Y']
        # Minimal paths: 1 period, 1 variable
        baseline_path = pd.DataFrame([[1.0]], columns=var_names)
        reform_path = pd.DataFrame([[1.02]], columns=var_names)
        fiscal_impact = pd.DataFrame({'horizon': ['impact'], 'revenue_change_pct': [0.02]})
        
        results = SimulationResults(
            name="Minimal Test",
            baseline_path=baseline_path,
            reform_path=reform_path,
            steady_state_baseline=SteadyState(),
            steady_state_reform=SteadyState(),
            welfare_change=0.01,
            fiscal_impact=fiscal_impact,
            transition_periods=1
        )
        
        # Should handle minimal data
        assert len(results.baseline_path) == 1
        assert len(results.reform_path) == 1
        assert len(results.baseline_path.columns) == 1
        
        # Should still be able to compute effects
        effects = results.compute_aggregate_effects(variables=['Y'])
        assert isinstance(effects, pd.DataFrame)
        assert 'Y' in effects.index
    
    def test_large_values_handling(self):
        """Test handling of large numerical values"""
        var_names = ['Y', 'C']
        # Large values
        large_baseline = pd.DataFrame(np.ones((10, 2)) * 1e6, columns=var_names)
        large_reform = pd.DataFrame(np.ones((10, 2)) * 1.1e6, columns=var_names)
        
        fiscal_impact = pd.DataFrame({'horizon': ['impact'], 'revenue_change_pct': [10.0]})
        
        results = SimulationResults(
            name="Large Values Test",
            baseline_path=large_baseline,
            reform_path=large_reform,
            steady_state_baseline=SteadyState(),
            steady_state_reform=SteadyState(),
            welfare_change=100.0,  # Large welfare change
            fiscal_impact=fiscal_impact,
            transition_periods=2
        )
        
        # Should handle large values
        assert np.all(np.isfinite(results.baseline_path.values))
        assert np.all(np.isfinite(results.reform_path.values))
        assert np.isfinite(results.welfare_change)
        
        # Should still compute effects correctly
        effects = results.compute_aggregate_effects(variables=['Y'])
        y_pct_change = effects.loc['Y', '% Change']
        assert abs(y_pct_change - 10.0) < 1e-6, "Should correctly compute 10% change"


if __name__ == "__main__":
    pytest.main([__file__])