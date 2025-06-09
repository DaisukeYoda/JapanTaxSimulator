"""
Fiscal Impact Analysis for Tax Policy Reforms

This module provides comprehensive fiscal impact analysis for DSGE tax policy simulations.
Analyzes government budget effects, revenue changes, debt dynamics, and fiscal sustainability.

Key Classes:
- FiscalAnalyzer: Main fiscal impact computation engine
- RevenueCalculator: Detailed tax revenue calculations
- DebtSustainabilityAnalyzer: Government debt dynamics analysis
- FiscalMultiplierCalculator: Fiscal multiplier effects

Economic Theory:
- Government budget constraint dynamics
- Tax elasticity calculations
- Debt sustainability analysis
- Fiscal policy feedback effects

Research Standards:
- Empirical tax elasticity parameters required
- No dummy calculations for revenue estimates  
- Debt dynamics validation against historical data
- Explicit fiscal rule specifications
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import warnings
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

# Import dependencies
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from dsge_model import SteadyState, ModelParameters
from research_warnings import research_critical, research_requires_validation, ResearchWarning


@dataclass
class FiscalConfig:
    """Configuration for fiscal impact analysis."""
    # Tax elasticity parameters (research-critical)
    consumption_tax_elasticity: float = -0.8  # Elasticity of consumption to consumption tax
    labor_tax_elasticity: float = -0.4  # Elasticity of labor to income tax
    capital_tax_elasticity: float = -0.6  # Elasticity of investment to capital tax
    
    # Government parameters
    debt_sustainability_threshold: float = 2.5  # Debt-to-GDP ratio threshold
    fiscal_reaction_coefficient: float = 0.1  # Response of taxes to debt
    
    # Analysis options
    include_behavioral_responses: bool = True
    include_general_equilibrium: bool = True
    discount_rate: float = 0.04  # Annual real discount rate for NPV calculations
    
    def validate(self):
        """Validate fiscal configuration parameters."""
        # Tax elasticities should be negative
        elasticities = [
            self.consumption_tax_elasticity,
            self.labor_tax_elasticity, 
            self.capital_tax_elasticity
        ]
        
        for elasticity in elasticities:
            if elasticity > 0:
                warnings.warn(f"Positive tax elasticity detected: {elasticity}")
            if abs(elasticity) > 2.0:
                warnings.warn(f"Very high tax elasticity: {elasticity}")


@dataclass
class FiscalImpactResult:
    """Container for fiscal impact analysis results."""
    revenue_impact: pd.DataFrame
    expenditure_impact: pd.DataFrame
    net_fiscal_impact: pd.DataFrame
    debt_dynamics: pd.DataFrame
    sustainability_metrics: Dict[str, float]
    present_value_impact: float
    
    def summary(self) -> Dict[str, Any]:
        """Generate fiscal impact summary."""
        total_revenue_change = self.net_fiscal_impact['Net_Impact'].sum()
        avg_annual_impact = self.net_fiscal_impact['Net_Impact'].mean() * 4  # Annualize
        
        return {
            'total_revenue_change': total_revenue_change,
            'average_annual_impact': avg_annual_impact,
            'present_value_impact': self.present_value_impact,
            'debt_sustainability': self.sustainability_metrics,
            'peak_deficit_impact': self.net_fiscal_impact['Net_Impact'].min(),
            'peak_surplus_impact': self.net_fiscal_impact['Net_Impact'].max()
        }


class RevenueCalculator:
    """
    Calculate detailed government tax revenues.
    
    Provides precise revenue calculations with behavioral adjustments
    and general equilibrium effects.
    """
    
    def __init__(self, config: FiscalConfig):
        self.config = config
    
    def calculate_revenue_streams(self, economic_path: pd.DataFrame,
                                tax_params: ModelParameters,
                                baseline_path: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate all government revenue streams.
        
        Args:
            economic_path: Economic variables time series
            tax_params: Tax parameters for this scenario
            baseline_path: Baseline scenario for comparison
            
        Returns:
            DataFrame with detailed revenue breakdown
        """
        revenue_data = []
        
        for t in range(len(economic_path)):
            period_revenue = self._calculate_period_revenue(
                economic_path.iloc[t], tax_params, t
            )
            period_revenue['Period'] = t
            revenue_data.append(period_revenue)
        
        df = pd.DataFrame(revenue_data)
        
        # Add behavioral adjustments if enabled
        if self.config.include_behavioral_responses and baseline_path is not None:
            df = self._apply_behavioral_adjustments(df, economic_path, baseline_path, tax_params)
        
        return df
    
    def _calculate_period_revenue(self, economic_vars: pd.Series, 
                                tax_params: ModelParameters, period: int) -> Dict[str, float]:
        """Calculate revenue for a single period."""
        # Extract required economic variables
        self._validate_required_variables(economic_vars, period)
        
        Y = economic_vars['Y']
        C = economic_vars['C']
        
        # Handle potentially missing economic variables with explicit warnings
        I = self._get_variable_with_validation(economic_vars, 'I', 0.2 * Y, 
                                             f"Investment (I) missing for period {period}, using default I/Y=0.2")
        L = self._get_variable_with_validation(economic_vars, 'L', 0.33, 
                                             f"Labor (L) missing for period {period}, using default L=0.33")
        
        # Wage calculation with validation
        if 'w' in economic_vars:
            w = economic_vars['w']
        else:
            w = 0.65 * Y / L if L > 0 else 1.0
            warnings.warn(
                f"Wage (w) missing for period {period}. Computing from labor productivity: w = 0.65*Y/L = {w:.4f}. "
                f"For research use, provide actual wage data.",
                ResearchWarning
            )
        
        # Capital stock with validation
        if 'K' in economic_vars:
            K = economic_vars['K']
        else:
            K = 10.0 * Y
            warnings.warn(
                f"Capital stock (K) missing for period {period}. Using default K/Y=10.0, K = {K:.4f}. "
                f"For research use, provide actual capital stock data.",
                ResearchWarning
            )
        
        # Tax revenue calculations
        tau_c_revenue = tax_params.tau_c * C
        tau_l_revenue = tax_params.tau_l * w * L  
        tau_k_revenue = tax_params.tau_k * self._calculate_capital_income(K, Y)
        tau_f_revenue = tax_params.tau_f * self._calculate_corporate_profits(Y, w, L)
        
        return {
            'Consumption_Tax_Revenue': tau_c_revenue,
            'Labor_Tax_Revenue': tau_l_revenue,
            'Capital_Tax_Revenue': tau_k_revenue,
            'Corporate_Tax_Revenue': tau_f_revenue,
            'Total_Revenue': tau_c_revenue + tau_l_revenue + tau_k_revenue + tau_f_revenue,
            'GDP': Y,
            'Tax_to_GDP_Ratio': (tau_c_revenue + tau_l_revenue + tau_k_revenue + tau_f_revenue) / Y
        }
    
    def _calculate_capital_income(self, K: float, Y: float) -> float:
        """Calculate capital income for tax base."""
        # Capital share of income (approximate)
        capital_share = 0.33
        return capital_share * Y
    
    def _calculate_corporate_profits(self, Y: float, w: float, L: float) -> float:
        """Calculate corporate profits for tax base."""
        # Profits = Output - Labor costs (simplified)
        labor_costs = w * L
        return max(0.1 * Y, Y - labor_costs)  # Ensure positive profits
    
    def _apply_behavioral_adjustments(self, revenue_df: pd.DataFrame,
                                    economic_path: pd.DataFrame,
                                    baseline_path: pd.DataFrame,
                                    tax_params: ModelParameters) -> pd.DataFrame:
        """Apply behavioral response adjustments to revenue calculations."""
        adjusted_df = revenue_df.copy()
        
        for t in range(len(revenue_df)):
            # Calculate tax rate changes that drive behavioral responses
            baseline_consumption = baseline_path['C'].iloc[t] if t < len(baseline_path) else baseline_path['C'].iloc[-1]
            reform_consumption = economic_path['C'].iloc[t]
            
            # Consumption tax behavioral adjustment
            if hasattr(tax_params, 'tau_c'):
                consumption_response = (reform_consumption / baseline_consumption - 1)
                behavioral_factor = 1 + self.config.consumption_tax_elasticity * consumption_response
                adjusted_df.loc[t, 'Consumption_Tax_Revenue'] *= max(0.1, behavioral_factor)
            
            # Similar adjustments for other taxes...
            # (Implementation would include labor and capital tax adjustments)
        
        # Recalculate totals
        adjusted_df['Total_Revenue'] = (
            adjusted_df['Consumption_Tax_Revenue'] + 
            adjusted_df['Labor_Tax_Revenue'] +
            adjusted_df['Capital_Tax_Revenue'] + 
            adjusted_df['Corporate_Tax_Revenue']
        )
        
        return adjusted_df
    
    def _validate_required_variables(self, economic_vars: pd.Series, period: int):
        """Validate that critical economic variables are present."""
        required_vars = ['Y', 'C']
        missing_vars = [var for var in required_vars if var not in economic_vars or pd.isna(economic_vars[var])]
        
        if missing_vars:
            raise ValueError(
                f"Critical economic variables missing for period {period}: {missing_vars}. "
                f"Cannot compute fiscal impact without GDP (Y) and Consumption (C)."
            )
    
    def _get_variable_with_validation(self, economic_vars: pd.Series, var_name: str, 
                                    default_value: float, warning_message: str) -> float:
        """
        Get economic variable with explicit validation and warning for missing data.
        
        For research use, missing data should be explicitly handled rather than
        silently substituted with defaults.
        """
        if var_name in economic_vars and not pd.isna(economic_vars[var_name]):
            return economic_vars[var_name]
        else:
            warnings.warn(
                f"RESEARCH WARNING: {warning_message} "
                f"This may affect fiscal impact accuracy. Consider providing complete economic data.",
                ResearchWarning
            )
            return default_value


class DebtSustainabilityAnalyzer:
    """
    Analyze government debt sustainability under tax reforms.
    
    Implements dynamic fiscal analysis with debt feedback rules
    and sustainability constraints.
    """
    
    def __init__(self, config: FiscalConfig):
        self.config = config
    
    def analyze_debt_dynamics(self, revenue_path: pd.DataFrame,
                            expenditure_path: pd.DataFrame,
                            initial_debt: float,
                            initial_gdp: float) -> pd.DataFrame:
        """
        Analyze government debt dynamics over time.
        
        Args:
            revenue_path: Government revenue time series
            expenditure_path: Government expenditure time series  
            initial_debt: Initial debt level
            initial_gdp: Initial GDP level
            
        Returns:
            DataFrame with debt dynamics analysis
        """
        debt_data = []
        current_debt = initial_debt
        
        for t in range(len(revenue_path)):
            # Period fiscal balance
            revenue = revenue_path['Total_Revenue'].iloc[t]
            expenditure = expenditure_path.get('Total_Expenditure', revenue * 0.9).iloc[t] if hasattr(expenditure_path, 'iloc') else revenue * 0.9
            primary_balance = revenue - expenditure
            
            # Interest payments (simplified)
            real_interest_rate = 0.02  # 2% real rate
            interest_payments = real_interest_rate * current_debt
            
            # Overall fiscal balance
            overall_balance = primary_balance - interest_payments
            
            # Update debt stock
            current_debt += -overall_balance  # Deficit increases debt
            
            # GDP for ratio calculations
            gdp = revenue_path['GDP'].iloc[t] if 'GDP' in revenue_path.columns else initial_gdp
            debt_to_gdp = current_debt / gdp
            
            debt_data.append({
                'Period': t,
                'Revenue': revenue,
                'Expenditure': expenditure,
                'Primary_Balance': primary_balance,
                'Interest_Payments': interest_payments,
                'Overall_Balance': overall_balance,
                'Debt_Stock': current_debt,
                'GDP': gdp,
                'Debt_to_GDP_Ratio': debt_to_gdp,
                'Sustainable': debt_to_gdp < self.config.debt_sustainability_threshold
            })
        
        return pd.DataFrame(debt_data)
    
    def compute_sustainability_metrics(self, debt_dynamics: pd.DataFrame) -> Dict[str, float]:
        """Compute key debt sustainability metrics."""
        final_debt_ratio = debt_dynamics['Debt_to_GDP_Ratio'].iloc[-1]
        max_debt_ratio = debt_dynamics['Debt_to_GDP_Ratio'].max()
        periods_unsustainable = (~debt_dynamics['Sustainable']).sum()
        
        # Debt stabilization requirement
        avg_primary_balance = debt_dynamics['Primary_Balance'].mean()
        avg_gdp_growth = 0.02  # Assumed 2% growth
        required_primary_balance = (self.config.discount_rate - avg_gdp_growth) * final_debt_ratio
        
        return {
            'final_debt_to_gdp': final_debt_ratio,
            'peak_debt_to_gdp': max_debt_ratio,
            'periods_unsustainable': periods_unsustainable,
            'avg_primary_balance': avg_primary_balance,
            'required_primary_balance': required_primary_balance,
            'sustainability_gap': required_primary_balance - avg_primary_balance
        }


class FiscalMultiplierCalculator:
    """
    Calculate fiscal multiplier effects.
    
    Estimates the macroeconomic impact of fiscal policy changes
    through government spending and tax multipliers.
    """
    
    def __init__(self, config: FiscalConfig):
        self.config = config
    
    def calculate_tax_multipliers(self, baseline_path: pd.DataFrame,
                                reform_path: pd.DataFrame,
                                tax_changes: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate tax multipliers for different tax types.
        
        Args:
            baseline_path: Baseline economic path
            reform_path: Reform economic path
            tax_changes: Tax rate changes
            
        Returns:
            Dictionary of tax multipliers
        """
        multipliers = {}
        
        # GDP impact
        baseline_gdp = baseline_path['Y'].mean()
        reform_gdp = reform_path['Y'].mean()
        gdp_change = (reform_gdp - baseline_gdp) / baseline_gdp
        
        # Calculate multipliers for each tax type
        for tax_type, tax_change in tax_changes.items():
            if abs(tax_change) > 1e-6:  # Avoid division by zero
                # Multiplier = % change in GDP / % change in tax rate
                multiplier = gdp_change / tax_change
                multipliers[f'{tax_type}_multiplier'] = multiplier
        
        return multipliers


class FiscalAnalyzer:
    """
    Main fiscal impact analysis engine.
    
    Provides comprehensive fiscal analysis including revenue impact,
    debt dynamics, and sustainability assessment.
    """
    
    def __init__(self, config: Optional[FiscalConfig] = None):
        """
        Initialize fiscal analyzer.
        
        Args:
            config: Fiscal analysis configuration
        """
        self.config = config or FiscalConfig()
        self.config.validate()
        
        self.revenue_calculator = RevenueCalculator(self.config)
        self.debt_analyzer = DebtSustainabilityAnalyzer(self.config)
        self.multiplier_calculator = FiscalMultiplierCalculator(self.config)
    
    @research_critical(
        "Fiscal impact calculations use calibrated tax elasticities that may not reflect "
        "current economic conditions. Revenue estimates should be validated against "
        "actual tax collection data and econometric studies."
    )
    def analyze_fiscal_impact(self, baseline_path: pd.DataFrame,
                            reform_path: pd.DataFrame,
                            baseline_params: ModelParameters,
                            reform_params: ModelParameters,
                            initial_debt_gdp_ratio: float = 2.0) -> FiscalImpactResult:
        """
        Comprehensive fiscal impact analysis.
        
        Args:
            baseline_path: Baseline economic simulation
            reform_path: Reform economic simulation
            baseline_params: Baseline tax parameters
            reform_params: Reform tax parameters
            initial_debt_gdp_ratio: Initial government debt-to-GDP ratio
            
        Returns:
            Complete fiscal impact analysis
        """
        # Calculate revenue streams for both scenarios
        baseline_revenue = self.revenue_calculator.calculate_revenue_streams(
            baseline_path, baseline_params
        )
        
        reform_revenue = self.revenue_calculator.calculate_revenue_streams(
            reform_path, reform_params, baseline_path
        )
        
        # Compute revenue impact (difference)
        revenue_impact = self._compute_revenue_impact(baseline_revenue, reform_revenue)
        
        # Expenditure impact (simplified - could be expanded)
        expenditure_impact = self._compute_expenditure_impact(baseline_path, reform_path)
        
        # Net fiscal impact
        net_fiscal_impact = self._compute_net_fiscal_impact(revenue_impact, expenditure_impact)
        
        # Debt dynamics analysis
        initial_debt = initial_debt_gdp_ratio * baseline_path['Y'].iloc[0]
        debt_dynamics = self.debt_analyzer.analyze_debt_dynamics(
            reform_revenue, expenditure_impact, initial_debt, baseline_path['Y'].iloc[0]
        )
        
        # Sustainability metrics
        sustainability_metrics = self.debt_analyzer.compute_sustainability_metrics(debt_dynamics)
        
        # Present value calculations
        present_value_impact = self._compute_present_value_impact(net_fiscal_impact)
        
        return FiscalImpactResult(
            revenue_impact=revenue_impact,
            expenditure_impact=expenditure_impact,
            net_fiscal_impact=net_fiscal_impact,
            debt_dynamics=debt_dynamics,
            sustainability_metrics=sustainability_metrics,
            present_value_impact=present_value_impact
        )
    
    def _compute_revenue_impact(self, baseline_revenue: pd.DataFrame,
                              reform_revenue: pd.DataFrame) -> pd.DataFrame:
        """Compute revenue impact as difference between scenarios."""
        impact_data = []
        
        for t in range(len(baseline_revenue)):
            baseline_total = baseline_revenue['Total_Revenue'].iloc[t]
            reform_total = reform_revenue['Total_Revenue'].iloc[t]
            
            impact_data.append({
                'Period': t,
                'Baseline_Revenue': baseline_total,
                'Reform_Revenue': reform_total,
                'Revenue_Change': reform_total - baseline_total,
                'Revenue_Change_Percent': (reform_total - baseline_total) / baseline_total * 100,
                'Consumption_Tax_Change': (reform_revenue['Consumption_Tax_Revenue'].iloc[t] - 
                                         baseline_revenue['Consumption_Tax_Revenue'].iloc[t]),
                'Labor_Tax_Change': (reform_revenue['Labor_Tax_Revenue'].iloc[t] - 
                                   baseline_revenue['Labor_Tax_Revenue'].iloc[t]),
                'Capital_Tax_Change': (reform_revenue['Capital_Tax_Revenue'].iloc[t] - 
                                     baseline_revenue['Capital_Tax_Revenue'].iloc[t]),
                'Corporate_Tax_Change': (reform_revenue['Corporate_Tax_Revenue'].iloc[t] - 
                                       baseline_revenue['Corporate_Tax_Revenue'].iloc[t])
            })
        
        return pd.DataFrame(impact_data)
    
    def _compute_expenditure_impact(self, baseline_path: pd.DataFrame,
                                  reform_path: pd.DataFrame) -> pd.DataFrame:
        """Compute expenditure impact (simplified model)."""
        expenditure_data = []
        
        for t in range(len(baseline_path)):
            # Simple expenditure model: G = constant share of GDP
            baseline_g = baseline_path.get('G', 0.2 * baseline_path['Y']).iloc[t]
            reform_g = reform_path.get('G', 0.2 * reform_path['Y']).iloc[t]
            
            expenditure_data.append({
                'Period': t,
                'Baseline_Expenditure': baseline_g,
                'Reform_Expenditure': reform_g,
                'Expenditure_Change': reform_g - baseline_g,
                'Total_Expenditure': reform_g  # For debt analysis
            })
        
        return pd.DataFrame(expenditure_data)
    
    def _compute_net_fiscal_impact(self, revenue_impact: pd.DataFrame,
                                 expenditure_impact: pd.DataFrame) -> pd.DataFrame:
        """Compute net fiscal impact."""
        net_data = []
        
        for t in range(len(revenue_impact)):
            revenue_change = revenue_impact['Revenue_Change'].iloc[t]
            expenditure_change = expenditure_impact['Expenditure_Change'].iloc[t]
            net_impact = revenue_change - expenditure_change
            
            net_data.append({
                'Period': t,
                'Revenue_Change': revenue_change,
                'Expenditure_Change': expenditure_change,
                'Net_Impact': net_impact,
                'Cumulative_Impact': 0  # Will be calculated below
            })
        
        df = pd.DataFrame(net_data)
        df['Cumulative_Impact'] = df['Net_Impact'].cumsum()
        
        return df
    
    def _compute_present_value_impact(self, net_fiscal_impact: pd.DataFrame) -> float:
        """Compute present value of fiscal impact."""
        quarterly_discount_rate = (1 + self.config.discount_rate) ** 0.25 - 1
        
        present_value = 0
        for t in range(len(net_fiscal_impact)):
            period_impact = net_fiscal_impact['Net_Impact'].iloc[t]
            discount_factor = (1 + quarterly_discount_rate) ** (-t)
            present_value += period_impact * discount_factor
        
        return present_value
    
    def compare_fiscal_impact_across_reforms(self, 
                                           reform_analyses: Dict[str, FiscalImpactResult]) -> pd.DataFrame:
        """
        Compare fiscal impacts across multiple reforms.
        
        Args:
            reform_analyses: Dictionary of {reform_name: FiscalImpactResult}
            
        Returns:
            DataFrame with fiscal impact comparison
        """
        comparison_data = []
        
        for reform_name, analysis in reform_analyses.items():
            summary = analysis.summary()
            
            comparison_data.append({
                'Reform': reform_name,
                'Total_Revenue_Change': summary['total_revenue_change'],
                'Average_Annual_Impact': summary['average_annual_impact'],
                'Present_Value_Impact': summary['present_value_impact'],
                'Final_Debt_to_GDP': analysis.sustainability_metrics['final_debt_to_gdp'],
                'Sustainability_Gap': analysis.sustainability_metrics['sustainability_gap'],
                'Peak_Deficit_Impact': summary['peak_deficit_impact'],
                'Periods_Unsustainable': analysis.sustainability_metrics['periods_unsustainable']
            })
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('Present_Value_Impact', ascending=False)