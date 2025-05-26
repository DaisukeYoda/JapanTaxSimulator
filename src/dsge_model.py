"""
DSGE Model Base Classes and Utilities for Japan Tax Simulator

This module provides the base classes and utility functions for implementing
a Dynamic Stochastic General Equilibrium (DSGE) model tailored for analyzing
the impact of tax policy changes on the Japanese economy.
"""

import numpy as np
import pandas as pd
from scipy import optimize
from scipy import linalg
from typing import Dict, Tuple, List, Optional
import json
from dataclasses import dataclass, field


@dataclass
class ModelParameters:
    """Container for all model parameters"""
    # Household parameters
    beta: float = 0.99  # Discount factor
    sigma_c: float = 1.5  # Intertemporal elasticity of substitution
    sigma_l: float = 2.0  # Frisch elasticity of labor supply
    habit: float = 0.6  # Habit formation parameter
    chi: float = 3.0  # Labor disutility parameter
    
    # Firm parameters
    alpha: float = 0.33  # Capital share in production
    delta: float = 0.025  # Depreciation rate
    theta_p: float = 0.75  # Calvo price stickiness
    epsilon: float = 6.0  # Elasticity of substitution
    psi: float = 4.0  # Capital adjustment cost
    
    # Government parameters
    gy_ratio: float = 0.20  # Gov spending to GDP ratio
    by_ratio: float = 2.0  # Gov debt to annual GDP ratio
    rho_g: float = 0.9  # Gov spending persistence
    phi_b: float = 0.1  # Fiscal rule parameter
    
    # Monetary policy parameters
    phi_pi: float = 1.5  # Taylor rule inflation coefficient
    phi_y: float = 0.125  # Taylor rule output coefficient
    rho_r: float = 0.8  # Interest rate smoothing
    pi_target: float = 1.005  # Inflation target (quarterly)
    
    # Tax parameters
    tau_c: float = 0.10  # Consumption tax rate
    tau_l: float = 0.20  # Labor income tax rate
    tau_k: float = 0.25  # Capital income tax rate
    tau_f: float = 0.30  # Corporate tax rate
    
    # Shock parameters
    rho_a: float = 0.95  # TFP persistence
    sigma_a: float = 0.01  # TFP shock std dev
    sigma_g: float = 0.01  # Gov spending shock std dev
    sigma_r: float = 0.0025  # Monetary policy shock std dev
    
    # Calibration targets
    cy_ratio: float = 0.60  # Consumption to GDP ratio
    iy_ratio: float = 0.20  # Investment to GDP ratio
    ky_ratio: float = 10.0  # Capital to annual GDP ratio
    hours_steady: float = 0.33  # Steady state hours worked
    
    @classmethod
    def from_json(cls, filepath: str) -> 'ModelParameters':
        """Load parameters from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        params = cls()
        
        # Extract model parameters
        for section in data['model_parameters'].values():
            for key, value in section.items():
                if not key.startswith('comment_'):
                    setattr(params, key, value)
        
        # Extract tax parameters
        for key, value in data['tax_parameters']['baseline'].items():
            if not key.startswith('comment_'):
                setattr(params, key, value)
        
        # Extract calibration targets
        for key, value in data['calibration_targets'].items():
            if not key.startswith('comment_'):
                setattr(params, key, value)
        
        return params


@dataclass
class SteadyState:
    """Container for steady state values"""
    Y: float = 1.0  # Output
    C: float = 0.6  # Consumption
    I: float = 0.2  # Investment
    K: float = 10.0  # Capital stock
    L: float = 0.33  # Labor hours
    w: float = 2.0  # Real wage
    r: float = 0.04  # Real interest rate
    pi: float = 1.005  # Inflation
    R: float = 1.015  # Nominal interest rate
    G: float = 0.2  # Government spending
    B: float = 2.0  # Government debt
    T: float = 0.2  # Total tax revenue
    Tc: float = 0.06  # Consumption tax revenue
    Tl: float = 0.08  # Labor tax revenue
    Tk: float = 0.03  # Capital tax revenue
    Tf: float = 0.03  # Corporate tax revenue
    Lambda: float = 1.0  # Marginal utility of consumption
    mc: float = 0.833  # Marginal cost
    profit: float = 0.1  # Firm profits
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def to_vector(self, variables: List[str]) -> np.ndarray:
        """Convert selected variables to vector"""
        return np.array([getattr(self, var) for var in variables])


class DSGEModel:
    """Base class for DSGE model"""
    
    def __init__(self, params: ModelParameters):
        self.params = params
        self.steady_state = None
        self.endogenous_vars = [
            'Y', 'C', 'I', 'K', 'L', 'w', 'r', 'pi', 'R', 'G', 'B',
            'T', 'Tc', 'Tl', 'Tk', 'Tf', 'Lambda', 'mc', 'profit'
        ]
        self.K_idx = self.endogenous_vars.index('K')
        self.L_idx = self.endogenous_vars.index('L')
        self.exogenous_vars = ['a', 'g', 'eps_r']  # TFP, gov spending, monetary shock
        
    def household_foc(self, C, C_prev, L, Lambda, w, r, pi_next, params):
        """
        Household first-order conditions
        Returns: (euler_equation, labor_supply)
        """
        # Marginal utility of consumption with habit formation
        if params.habit > 0:
            Lambda_calc = (C - params.habit * C_prev) ** (-params.sigma_c)
        else:
            Lambda_calc = C ** (-params.sigma_c)
        
        # Euler equation
        euler = Lambda - params.beta * Lambda * (1 + r) / pi_next
        
        # Labor supply condition
        labor_supply = (w * (1 - params.tau_l) * Lambda - 
                       params.chi * L ** (1 / params.sigma_l))
        
        return euler, labor_supply, Lambda_calc - Lambda
    
    def firm_foc(self, Y, K, L, w, r, mc, pi, params):
        """
        Firm first-order conditions
        Returns: (production, labor_demand, capital_demand, pricing)
        """
        # Production function
        production = Y - K ** params.alpha * L ** (1 - params.alpha)
        
        # Labor demand
        labor_demand = w - mc * (1 - params.alpha) * Y / L
        
        # Capital demand (rental rate)
        capital_demand = r - mc * params.alpha * Y / K * (1 - params.tau_f)
        
        # Phillips curve (simplified)
        pricing = (pi - params.pi_target) - params.beta * (pi - params.pi_target) - \
                 ((1 - params.theta_p) * (1 - params.beta * params.theta_p) / 
                  params.theta_p) * (mc - (params.epsilon - 1) / params.epsilon)
        
        return production, labor_demand, capital_demand, pricing
    
    def government_budget(self, G, B, T, Tc, Tl, Tk, Tf, r, Y, C, L, K, w, profit, params):
        """
        Government budget constraint and fiscal rules
        """
        # Tax revenues
        Tc_calc = params.tau_c * C
        Tl_calc = params.tau_l * w * L
        Tk_calc = params.tau_k * r * K
        Tf_calc = params.tau_f * profit
        T_calc = Tc_calc + Tl_calc + Tk_calc + Tf_calc
        
        # Budget constraint
        budget = B - (1 + r) * B + G - T
        
        # Fiscal rule (adjust spending based on debt)
        fiscal_rule = G - params.gy_ratio * Y * (1 - params.phi_b * (B / Y - params.by_ratio))
        
        return (budget, fiscal_rule, T_calc - T, Tc_calc - Tc, 
                Tl_calc - Tl, Tk_calc - Tk, Tf_calc - Tf)
    
    def monetary_policy(self, R, pi, Y, Y_ss, params):
        """
        Taylor rule for monetary policy
        """
        taylor_rule = (R - params.pi_target * 
                      ((pi / params.pi_target) ** params.phi_pi * 
                       (Y / Y_ss) ** params.phi_y) ** (1 - params.rho_r) * 
                      R ** params.rho_r)
        
        return taylor_rule
    
    def market_clearing(self, Y, C, I, G, K, K_next, params):
        """
        Market clearing conditions
        """
        # Goods market
        goods_market = Y - C - I - G
        
        # Capital accumulation
        capital_acc = K_next - (1 - params.delta) * K - I
        
        # Investment adjustment costs (simplified)
        investment = I - params.delta * K
        
        return goods_market, capital_acc, investment
    
    def compute_steady_state(self, initial_guess: Optional[Dict] = None) -> SteadyState:
        """
        Compute the steady state of the model
        """
        params = self.params
        
        # Define the steady state system of equations
        def steady_state_system(x):
            # Unpack variables from x
            # Y, C, I, K_val, L_val, w, r, pi, R, G, B, T, Tc, Tl, Tk, Tf, Lambda, mc, profit = x
            
            # Exponentiate K and L
            K = np.exp(x[self.K_idx])
            L = np.exp(x[self.L_idx])
            
            # Assign other variables directly, K and L are handled
            # This manual unpacking is a bit verbose but clear.
            Y = x[0]
            C = x[1]
            I = x[2]
            # K is now from np.exp(x[self.K_idx])
            # L is now from np.exp(x[self.L_idx])
            w = x[5]
            r = x[6]
            pi = x[7]
            R = x[8]
            G = x[9]
            B = x[10]
            T = x[11]
            Tc = x[12]
            Tl = x[13]
            Tk = x[14]
            Tf = x[15]
            Lambda = x[16]
            mc = x[17]
            profit = x[18]

            # Set steady state values
            C_prev = C  # In steady state
            pi_next = pi
            K_next = K
            Y_ss = Y
            
            # Compute all first-order conditions
            euler, labor_supply, lambda_eq = self.household_foc(
                C, C_prev, L, Lambda, w, r, pi_next, params
            )
            
            production, labor_demand, capital_demand, pricing = self.firm_foc(
                Y, K, L, w, r, mc, pi, params
            )
            
            budget, fiscal_rule, *tax_eqs = self.government_budget(
                G, B, T, Tc, Tl, Tk, Tf, r, Y, C, L, K, w, profit, params
            )
            
            taylor = self.monetary_policy(R, pi, Y, Y_ss, params)
            
            goods_market, capital_acc, investment = self.market_clearing(
                Y, C, I, G, K, K_next, params
            )
            
            # Additional steady state conditions
            profit_eq = profit - (1 - mc) * Y
            real_rate = R / pi - (1 + r)
            
            # Stack all equations
            equations = [
                euler, labor_supply, lambda_eq,
                production, labor_demand, capital_demand, pricing,
                budget, fiscal_rule, *tax_eqs,
                taylor,
                goods_market, investment,
                profit_eq, real_rate
            ]
            
            return np.array(equations)
        
        # Initial guess
        if initial_guess is None:
            x0 = np.array([
                1.0,           # Y
                0.6,           # C
                0.2,           # I
                np.log(10.0),  # K (log-transformed)
                np.log(0.33),  # L (log-transformed)
                2.0,           # w
                0.04,          # r
                1.005,         # pi
                1.045,         # R
                0.2,           # G
                2.0,           # B
                0.2,           # T
                0.06,          # Tc
                0.08,          # Tl
                0.03,          # Tk
                0.03,          # Tf
                1.0,           # Lambda
                0.833,         # mc
                0.1            # profit
            ])
        else:
            x0_list = []
            for var_name in self.endogenous_vars:
                val = initial_guess.get(var_name, 1.0)
                if var_name == 'K' or var_name == 'L':
                    val = np.log(val)
                x0_list.append(val)
            x0 = np.array(x0_list)
        
        # Solve the system
        solution = optimize.fsolve(steady_state_system, x0, full_output=True)
        
        if solution[2] != 1:
            raise ValueError(f"Steady state computation failed: {solution[3]}")
        
        # Create steady state object
        ss_values = solution[0]
        steady_state = SteadyState()
        
        for i, var_name in enumerate(self.endogenous_vars):
            val_to_set = ss_values[i]
            if var_name == 'K' or var_name == 'L':
                val_to_set = np.exp(val_to_set)
            setattr(steady_state, var_name, val_to_set)
        
        self.steady_state = steady_state
        return steady_state
    
    def check_steady_state(self, ss: SteadyState) -> Dict[str, float]:
        """
        Check if computed steady state satisfies calibration targets
        """
        errors = {}
        
        # Check ratios
        errors['C/Y'] = ss.C / ss.Y - self.params.cy_ratio
        errors['I/Y'] = ss.I / ss.Y - self.params.iy_ratio
        errors['K/Y'] = ss.K / (4 * ss.Y) - self.params.ky_ratio  # Annual
        errors['Hours'] = ss.L - self.params.hours_steady
        
        # Check government ratios
        errors['G/Y'] = ss.G / ss.Y - self.params.gy_ratio
        errors['B/Y'] = ss.B / (4 * ss.Y) - self.params.by_ratio  # Annual
        
        # Check market clearing
        errors['Goods market'] = ss.Y - ss.C - ss.I - ss.G
        errors['Capital acc'] = ss.I - self.params.delta * ss.K
        
        return errors


def load_model(config_path: str) -> DSGEModel:
    """Load model with parameters from config file"""
    params = ModelParameters.from_json(config_path)
    return DSGEModel(params)
