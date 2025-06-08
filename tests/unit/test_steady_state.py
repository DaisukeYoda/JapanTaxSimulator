"""
Unit tests for SteadyState class from dsge_model module

Tests steady state initialization, validation, conversions, and economic relationships.
"""

import pytest
import numpy as np
from dataclasses import asdict

from src.dsge_model import SteadyState, ModelParameters


class TestSteadyStateBasic:
    """Test basic SteadyState functionality"""
    
    def test_default_initialization(self):
        """Test that SteadyState initializes with default values"""
        ss = SteadyState()
        
        # Test key macroeconomic variables
        assert ss.Y == 1.0
        assert ss.C == 0.6
        assert ss.I == 0.2
        assert ss.K == 10.0
        assert ss.L == 0.33
        
        # Test prices and rates
        assert ss.w == 2.0
        assert ss.Rk_gross == 0.065
        assert ss.r_net_real == 0.0101
        assert ss.pi_gross == 1.005
        assert ss.i_nominal_gross == 1.0151
        
        # Test government variables
        assert ss.G == 0.2
        assert ss.B_real == 2.0
        
        # Test firm variables
        assert ss.Lambda == 1.0
        assert ss.mc == 0.833
        assert ss.profit == 0.167
        
        # Test international variables
        assert ss.q == 1.0
        assert ss.b_star == 0.0
        assert ss.IM == 0.15
        assert ss.EX == 0.15
        assert ss.NX == 0.0
        assert ss.Y_star == 1.0
    
    def test_custom_initialization(self):
        """Test SteadyState initialization with custom values"""
        ss = SteadyState(
            Y=1.2,
            C=0.7,
            I=0.25,
            K=12.0,
            L=0.35
        )
        
        assert ss.Y == 1.2
        assert ss.C == 0.7
        assert ss.I == 0.25
        assert ss.K == 12.0
        assert ss.L == 0.35
        
        # Other values should remain at defaults
        assert ss.w == 2.0
        assert ss.pi_gross == 1.005
    
    def test_post_init_calculations(self):
        """Test that __post_init__ performs correct calculations"""
        ss = SteadyState(r_net_real=0.02)
        
        # Test R_star_gross_real calculation
        expected_R_star = 1 + ss.r_net_real
        assert ss.R_star_gross_real == expected_R_star
        
        # Test i_nominal_net calculation
        expected_i_net = ss.i_nominal_gross - 1
        assert ss.i_nominal_net == expected_i_net
        
        # Test default values set in post_init
        assert ss.tau_l_effective == 0.20
        assert ss.A_tfp == 1.0
        assert ss.T_transfer == 0.0


class TestSteadyStateValidation:
    """Test steady state validation and economic relationships"""
    
    def test_positive_values(self):
        """Test that key variables are positive"""
        ss = SteadyState()
        
        positive_vars = [
            'Y', 'C', 'I', 'K', 'L', 'w', 'Rk_gross', 'pi_gross', 
            'i_nominal_gross', 'G', 'B_real', 'Lambda', 'mc', 'q'
        ]
        
        for var_name in positive_vars:
            value = getattr(ss, var_name)
            assert value > 0, f"{var_name} should be positive, got {value}"
    
    def test_rates_bounds(self):
        """Test that rates are within reasonable bounds"""
        ss = SteadyState()
        
        # Real interest rate should be small and positive for typical calibrations
        assert 0 < ss.r_net_real < 0.1, f"Real rate should be reasonable, got {ss.r_net_real}"
        
        # Nominal interest rate should be higher than real rate
        assert ss.i_nominal_gross > 1 + ss.r_net_real, "Nominal rate should account for inflation"
        
        # Gross inflation should be close to 1 for low inflation
        assert 1.0 < ss.pi_gross < 1.1, f"Gross inflation should be reasonable, got {ss.pi_gross}"
        
        # Marginal cost should be less than 1 under monopolistic competition
        assert 0 < ss.mc < 1, f"Marginal cost should be between 0 and 1, got {ss.mc}"
    
    def test_accounting_identities(self):
        """Test basic accounting identities"""
        ss = SteadyState()
        
        # GDP identity: Y = C + I + G + NX (in steady state)
        gdp_components = ss.C + ss.I + ss.G + ss.NX
        assert abs(ss.Y - gdp_components) < 1e-10, f"GDP identity violated: Y={ss.Y}, C+I+G+NX={gdp_components}"
        
        # Trade balance: NX = EX - IM
        trade_balance = ss.EX - ss.IM
        assert abs(ss.NX - trade_balance) < 1e-10, f"Trade balance identity violated: NX={ss.NX}, EX-IM={trade_balance}"
    
    def test_tax_revenue_consistency(self):
        """Test that tax revenue components are reasonable"""
        ss = SteadyState()
        
        # All tax revenues should be non-negative
        assert ss.Tc >= 0, "Consumption tax revenue should be non-negative"
        assert ss.Tl >= 0, "Labor tax revenue should be non-negative"
        assert ss.Tk >= 0, "Capital tax revenue should be non-negative"
        assert ss.Tf >= 0, "Corporate tax revenue should be non-negative"
        
        # Total tax revenue should equal sum of components
        total_components = ss.Tc + ss.Tl + ss.Tk + ss.Tf
        assert abs(ss.T_total_revenue - total_components) < 1e-10, "Total tax revenue should equal sum of components"
    
    def test_production_function_consistency(self):
        """Test that production function variables are consistent"""
        ss = SteadyState()
        params = ModelParameters()
        
        # Test that implied TFP would be reasonable
        # Y = A * K^α * L^(1-α), so A = Y / (K^α * L^(1-α))
        if ss.K > 0 and ss.L > 0:
            implied_tfp = ss.Y / (ss.K**params.alpha * ss.L**(1-params.alpha))
            assert 0.5 < implied_tfp < 2.0, f"Implied TFP should be reasonable, got {implied_tfp}"


class TestSteadyStateConversions:
    """Test conversion and utility methods"""
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        ss = SteadyState()
        ss_dict = ss.to_dict()
        
        # Check that it returns a dictionary
        assert isinstance(ss_dict, dict)
        
        # Check that all non-private attributes are included
        ss_attrs = {k: v for k, v in asdict(ss).items() if not k.startswith('_')}
        assert len(ss_dict) == len(ss_attrs)
        
        # Check that values match
        for key, value in ss_attrs.items():
            assert key in ss_dict
            assert ss_dict[key] == value
    
    def test_to_dict_excludes_private_attributes(self):
        """Test that to_dict excludes private attributes"""
        ss = SteadyState()
        # Add a mock private attribute
        ss._private_attr = "should not appear"
        
        ss_dict = ss.to_dict()
        assert '_private_attr' not in ss_dict


class TestSteadyStateEconomicRatios:
    """Test economic ratios and relationships"""
    
    def test_default_ratios(self):
        """Test that default values give reasonable economic ratios"""
        ss = SteadyState()
        
        # Consumption-to-GDP ratio should be reasonable (typically 0.5-0.7)
        c_y_ratio = ss.C / ss.Y
        assert 0.4 < c_y_ratio < 0.8, f"C/Y ratio should be reasonable, got {c_y_ratio}"
        
        # Investment-to-GDP ratio should be reasonable (typically 0.15-0.25)
        i_y_ratio = ss.I / ss.Y
        assert 0.1 < i_y_ratio < 0.3, f"I/Y ratio should be reasonable, got {i_y_ratio}"
        
        # Government-to-GDP ratio should be reasonable (typically 0.15-0.25)
        g_y_ratio = ss.G / ss.Y
        assert 0.1 < g_y_ratio < 0.3, f"G/Y ratio should be reasonable, got {g_y_ratio}"
        
        # Capital-to-output ratio should be reasonable (annual basis, typically 2-12)
        k_y_annual_ratio = ss.K / (4 * ss.Y)  # Convert to annual
        assert 1 < k_y_annual_ratio < 15, f"K/Y annual ratio should be reasonable, got {k_y_annual_ratio}"
    
    def test_labor_market_ratios(self):
        """Test labor market relationships"""
        ss = SteadyState()
        
        # Hours worked should be reasonable fraction (typically 0.25-0.4)
        assert 0.2 < ss.L < 0.5, f"Labor hours should be reasonable fraction, got {ss.L}"
        
        # Wage should be reasonable relative to output per worker
        if ss.L > 0:
            output_per_worker = ss.Y / ss.L
            wage_to_productivity = ss.w / output_per_worker
            assert 0.4 < wage_to_productivity < 1.0, f"Wage-to-productivity ratio should be reasonable, got {wage_to_productivity}"
    
    def test_financial_ratios(self):
        """Test financial variables and ratios"""
        ss = SteadyState()
        
        # Debt-to-GDP ratio should be reasonable (quarterly basis)
        debt_gdp_ratio = ss.B_real / ss.Y
        assert 0.1 < debt_gdp_ratio < 4.0, f"Debt/GDP ratio should be reasonable, got {debt_gdp_ratio}"
        
        # Real interest rate should be positive but not too high
        assert 0.001 < ss.r_net_real < 0.05, f"Real interest rate should be reasonable, got {ss.r_net_real}"
        
        # Return on capital should be higher than risk-free rate
        assert ss.Rk_gross > ss.r_net_real, "Return on capital should exceed risk-free rate"


class TestSteadyStateEdgeCases:
    """Test edge cases and special scenarios"""
    
    def test_zero_values_handling(self):
        """Test handling of zero or near-zero values"""
        # Test with very small positive values
        ss = SteadyState(
            L=1e-6,
            K=1e-6
        )
        
        # Should not cause division by zero errors in calculations
        ss_dict = ss.to_dict()
        assert all(np.isfinite(v) for v in ss_dict.values() if isinstance(v, (int, float)))
    
    def test_extreme_values_handling(self):
        """Test handling of extreme but valid values"""
        ss = SteadyState(
            Y=100.0,
            C=60.0,
            I=20.0,  # Adjust I to match Y scale
            G=20.0,  # Adjust G to match Y scale
            K=1000.0
        )
        
        # Should maintain consistent relationships
        ss_dict = ss.to_dict()
        assert all(np.isfinite(v) for v in ss_dict.values() if isinstance(v, (int, float)))
        
        # Basic accounting should hold approximately (given the scale mismatch)
        gdp_error = abs(ss.Y - (ss.C + ss.I + ss.G + ss.NX))
        assert gdp_error < 1.0, f"GDP accounting error should be small relative to scale, got {gdp_error}"
    
    def test_attribute_access(self):
        """Test that all expected attributes can be accessed"""
        ss = SteadyState()
        
        # Get all attributes from the dataclass
        all_attrs = asdict(ss).keys()
        
        # Check that we can access all attributes without error
        for attr_name in all_attrs:
            value = getattr(ss, attr_name)
            assert value is not None or attr_name in ['T_transfer', 'b_star', 'NX'], f"Unexpected None value for {attr_name}"


if __name__ == "__main__":
    pytest.main([__file__])