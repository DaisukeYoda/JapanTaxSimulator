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
import sympy
from sympy import log, exp # Import specific sympy functions for convenience

@dataclass
class ModelParameters:
    """Container for all model parameters"""
    # Household parameters
    beta: float = 0.99; sigma_c: float = 1.5; sigma_l: float = 2.0
    habit: float = 0.6; chi: float = 3.0
    # Firm parameters
    alpha: float = 0.33; delta: float = 0.025; theta_p: float = 0.75
    epsilon: float = 6.0; psi: float = 4.0
    # Government parameters
    gy_ratio: float = 0.20; by_ratio: float = 2.0 # Target Debt-to-Annual-GDP ratio
    rho_g: float = 0.9; phi_b: float = 0.1; tau_l_ss: float = 0.20; tau_l: float = 0.20
    # Monetary policy parameters
    phi_pi: float = 1.5; phi_y: float = 0.125; rho_r: float = 0.8
    pi_target: float = 1.005 # Gross quarterly inflation target
    # Tax parameters
    tau_c: float = 0.10; tau_k: float = 0.25; tau_f: float = 0.30
    # Shock parameters
    rho_a: float = 0.95; sigma_a: float = 0.01; sigma_g: float = 0.01
    sigma_r: float = 0.0025; sigma_ystar: float = 0.01
    # International Parameters
    alpha_m: float = 0.15; eta_im: float = 1.0; alpha_x: float = 0.15
    phi_ex: float = 1.5; eta_ex: float = 1.0; ystar_ss: float = 1.0
    rho_ystar: float = 0.90
    # Calibration targets
    cy_ratio: float = 0.60; iy_ratio: float = 0.20; ky_ratio: float = 10.0 # K/Y annual
    hours_steady: float = 0.33; nx_y_ratio_target: float = 0.0
    b_star_target_level: float = 0.0

    @classmethod
    def from_json(cls, filepath: str) -> 'ModelParameters':
        with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
        params = cls()
        params.tau_l_ss = data.get('tax_parameters', {}).get('baseline', {}).get('tau_l', params.tau_l_ss)
        params.tau_l = params.tau_l_ss  # Set tau_l to match tau_l_ss initially
        param_sections_data = data['model_parameters']
        for section_key in param_sections_data: 
            for k, v in param_sections_data[section_key].items():
                if not k.startswith('comment_') and hasattr(params, k): setattr(params, k, v)
        tax_base = data.get('tax_parameters', {}).get('baseline', {})
        for k, v in tax_base.items():
            if not k.startswith('comment_') and hasattr(params, k): setattr(params, k, v)
        # Ensure tau_l is set from JSON if present, otherwise use tau_l_ss
        if 'tau_l' in tax_base:
            params.tau_l = tax_base['tau_l']
        calib_targets = data.get('calibration_targets', {})
        for k, v in calib_targets.items():
            if not k.startswith('comment_') and hasattr(params, k): setattr(params, k, v)
        return params

@dataclass
class SteadyState: 
    Y: float=1.0; C: float=0.6; I: float=0.2; K: float=10.0; L: float=0.33
    w: float=2.0; Rk_gross: float=0.065; r_net_real: float=0.0101
    pi_gross: float=1.005; i_nominal_gross: float=1.0151 
    G: float=0.2; B_real: float=2.0 
    Lambda: float=1.0; mc: float=0.833; profit: float=0.167
    q: float=1.0; b_star: float=0.0
    IM: float=0.15; EX: float=0.15; A_dom: float=0.8
    NX: float=0.0; Y_star: float=1.0
    R_star_gross_real: float=1.0101 
    Tc: float=0.06; Tl: float=0.066; Tk: float=0.016; Tf: float=0.05; T_total_revenue: float=0.192
    T_transfer: float=0.0; tau_l_effective: float=0.20; A_tfp: float=1.0
    i_nominal_net: float=0.0151

    def to_dict(self) -> Dict[str, float]: return {k:v for k,v in self.__dict__.items() if not k.startswith('_')}
    def __post_init__(self): 
        self.R_star_gross_real=(1+self.r_net_real) 
        self.i_nominal_net=self.i_nominal_gross-1
        mp=ModelParameters(); self.tau_l_effective=mp.tau_l_ss
        self.A_tfp=1.0; self.T_transfer=0.0


class DSGEModel:
    def __init__(self, params: ModelParameters):
        self.params = params
        self.steady_state: Optional[SteadyState] = None
        self.endogenous_vars_solve = [ # For steady-state solver
            'Y','C','I','K','L','w','Rk_gross','r_net_real','pi_gross', 
            'i_nominal_gross','G','B_real','Lambda','mc','profit','q', 
            'b_star','IM','EX'] 
        self.log_vars_indices = {} 
        try: self.log_vars_indices = {'K':self.endogenous_vars_solve.index('K'), 'L':self.endogenous_vars_solve.index('L')}
        except ValueError: pass 
        self.exogenous_shocks_sym_names = ['eps_a', 'eps_g', 'eps_r', 'eps_ystar'] 

    def _sym(self, name_base: str, lags: int=0, leads: int=0):
        suff = ['', *[f'_tm{i+1}' for i in range(lags)], *[f'_tp{i+1}' for i in range(leads)]]
        syms = [sympy.Symbol(f"{name_base}{s}") for s in suff]
        return syms[0] if lags==0 and leads==0 else tuple(syms)

    def get_equations_for_steady_state(self, x_solve: np.ndarray) -> np.ndarray:
        params=self.params; vars_dict={}
        for i,var_name in enumerate(self.endogenous_vars_solve):
            val=x_solve[i]
            if var_name in self.log_vars_indices:vars_dict[var_name]=np.exp(val)
            else:vars_dict[var_name]=val
        Y=vars_dict['Y'];C=vars_dict['C'];I=vars_dict['I'];K=vars_dict['K'];L=vars_dict['L']
        w=vars_dict['w'];Rk_gross=vars_dict['Rk_gross'];r_net_real=vars_dict['r_net_real']
        pi_gross=vars_dict['pi_gross'];i_nominal_gross=vars_dict['i_nominal_gross'];G=vars_dict['G']
        B_real=vars_dict['B_real'];Lambda=vars_dict['Lambda'];mc=vars_dict['mc'];profit=vars_dict['profit']
        q_val=vars_dict['q'];b_star=vars_dict['b_star'];IM=vars_dict['IM'];EX=vars_dict['EX'] 
        A_val=C+I+G;Y_star_val=params.ystar_ss
        by_target_q=params.by_ratio/4  # Convert annual debt/GDP ratio to quarterly 
        # Calculate effective labor tax rate with bounds to prevent unrealistic values
        debt_feedback = params.phi_b*((B_real/(Y if Y!=0 else 1e-6))-by_target_q)
        tau_l_effective_ss = max(0.05, min(0.8, params.tau_l_ss + debt_feedback))  # Bounded between 5% and 80%
        Tc_val=params.tau_c*C;Tl_val=tau_l_effective_ss*w*L;Tk_val=params.tau_k*Rk_gross*K
        Tf_val=params.tau_f*profit;T_val=Tc_val+Tl_val+Tk_val+Tf_val
        eqns=[(1-params.beta*params.habit)/(C*(1-params.habit))-Lambda*(1+params.tau_c),
              params.chi*L**(1/params.sigma_l)-Lambda*(1-tau_l_effective_ss)*w/(1+params.tau_c),
              Lambda-params.beta*Lambda*(1+r_net_real)/pi_gross,Y-K**params.alpha*L**(1-params.alpha),
              w-mc*(1-params.alpha)*Y/L if L!=0 else w-mc*(1-params.alpha)*Y/1e-6,
              Rk_gross-mc*params.alpha*Y/K if K!=0 else Rk_gross-mc*params.alpha*Y/1e-6,
              pi_gross-params.pi_target,mc-(params.epsilon-1)/params.epsilon,
              G+r_net_real*B_real-T_val, 
              G-params.gy_ratio*Y*(1-params.phi_b*((B_real/(Y if Y!=0 else 1e-6))-by_target_q)),
              i_nominal_gross-params.pi_target*(pi_gross/params.pi_target)**params.phi_pi,
              (1+r_net_real)-i_nominal_gross/pi_gross,I-params.delta*K,
              (1-params.tau_k)*Rk_gross-params.delta-r_net_real,profit-(1-mc)*Y,
              IM-params.alpha_m*A_val*q_val**(-params.eta_im),
              EX-params.alpha_x*(Y_star_val**params.phi_ex)*q_val**params.eta_ex,
              b_star-params.b_star_target_level,Y-(C+I+G+q_val*b_star*(1-(1/params.beta)))]
        return np.array(eqns)

    def _compute_tax_adjusted_initial_guess(self, baseline_ss: Optional[SteadyState] = None) -> Dict[str, float]:
        """
        Compute initial guess adjusted for tax parameter changes from a baseline steady state
        """
        if baseline_ss is None:
            return {}
        
        # Create a reference ModelParameters with baseline tax rates for comparison
        ref_params = ModelParameters()
        ref_params.tau_c = 0.10  # Baseline consumption tax
        ref_params.tau_l = 0.20  # Baseline labor tax  
        ref_params.tau_k = 0.25  # Baseline capital tax
        ref_params.tau_f = 0.30  # Baseline corporate tax
        
        params = self.params
        initial_guess = {}
        
        # Calculate tax wedge adjustments
        consumption_tax_ratio = (1 + ref_params.tau_c) / (1 + params.tau_c)
        labor_tax_ratio = (1 - ref_params.tau_l) / (1 - params.tau_l)
        capital_tax_ratio = (1 - ref_params.tau_k) / (1 - params.tau_k)
        
        for var in self.endogenous_vars_solve:
            baseline_val = getattr(baseline_ss, var)
            
            if var == 'C':
                # Consumption responds to consumption tax changes
                # Higher tau_c reduces consumption
                initial_guess[var] = baseline_val * (consumption_tax_ratio ** 0.5)
                
            elif var == 'L':
                # Labor supply responds to labor tax changes
                # Higher tau_l reduces labor supply
                initial_guess[var] = baseline_val * (labor_tax_ratio ** 0.3)
                
            elif var == 'I':
                # Investment responds to capital tax changes
                # Higher tau_k reduces investment
                initial_guess[var] = baseline_val * (capital_tax_ratio ** 0.4)
                
            elif var == 'K':
                # Capital stock adjusts to investment changes (but slowly)
                initial_guess[var] = baseline_val * (capital_tax_ratio ** 0.2)
                
            elif var == 'w':
                # Wage adjusts to labor supply changes
                initial_guess[var] = baseline_val * (labor_tax_ratio ** -0.2)
                
            elif var == 'Rk_gross':
                # Gross return on capital must adjust for tax changes
                net_return = baseline_ss.Rk_gross * (1 - ref_params.tau_k)
                initial_guess[var] = net_return / max(1 - params.tau_k, 0.1)
                
            elif var == 'Lambda':
                # Marginal utility of consumption (inverse relationship with C)
                initial_guess[var] = baseline_val / (consumption_tax_ratio ** 0.5)
                
            elif var == 'G':
                # Government spending adjusts partially to revenue changes
                baseline_revenue = (ref_params.tau_c * baseline_ss.C + 
                                  ref_params.tau_l * baseline_ss.w * baseline_ss.L +
                                  ref_params.tau_k * baseline_ss.Rk_gross * baseline_ss.K +
                                  ref_params.tau_f * baseline_ss.profit)
                new_revenue_est = (params.tau_c * initial_guess.get('C', baseline_ss.C) +
                                 params.tau_l * initial_guess.get('w', baseline_ss.w) * initial_guess.get('L', baseline_ss.L) +
                                 params.tau_k * initial_guess.get('Rk_gross', baseline_ss.Rk_gross) * initial_guess.get('K', baseline_ss.K) +
                                 params.tau_f * baseline_ss.profit)
                revenue_change = new_revenue_est - baseline_revenue
                initial_guess[var] = baseline_val + 0.3 * revenue_change  # 30% of extra revenue to spending
                
            elif var == 'B_real':
                # Government debt adjusts to revenue changes (opposite direction)
                baseline_revenue = (ref_params.tau_c * baseline_ss.C + 
                                  ref_params.tau_l * baseline_ss.w * baseline_ss.L +
                                  ref_params.tau_k * baseline_ss.Rk_gross * baseline_ss.K +
                                  ref_params.tau_f * baseline_ss.profit)
                new_revenue_est = (params.tau_c * initial_guess.get('C', baseline_ss.C) +
                                 params.tau_l * initial_guess.get('w', baseline_ss.w) * initial_guess.get('L', baseline_ss.L) +
                                 params.tau_k * initial_guess.get('Rk_gross', baseline_ss.Rk_gross) * initial_guess.get('K', baseline_ss.K) +
                                 params.tau_f * baseline_ss.profit)
                revenue_change = new_revenue_est - baseline_revenue
                debt_adjustment = -2.0 * revenue_change  # Debt falls with higher revenue
                initial_guess[var] = max(baseline_val + debt_adjustment, 0.1 * baseline_val)
                
            elif var == 'Y':
                # Output adjusts to changes in inputs (slower adjustment)
                K_adj = initial_guess.get('K', baseline_val) / baseline_ss.K
                L_adj = initial_guess.get('L', baseline_val) / baseline_ss.L
                Y_adj = (K_adj ** params.alpha) * (L_adj ** (1 - params.alpha))
                initial_guess[var] = baseline_val * Y_adj
                
            else:
                # For other variables, use baseline values with small random perturbation
                initial_guess[var] = baseline_val * (1 + 0.01 * np.random.normal())
        
        return initial_guess

    def compute_steady_state(self, initial_guess_dict: Optional[Dict]=None, baseline_ss: Optional[SteadyState] = None) -> SteadyState:
        params=self.params;ss_defaults=SteadyState();ss_defaults.tau_l_effective=params.tau_l_ss
        ss_defaults.pi_gross=params.pi_target;ss_defaults.r_net_real=(1/params.beta)-1 
        ss_defaults.i_nominal_net=(1+ss_defaults.r_net_real)*ss_defaults.pi_gross-1;ss_defaults.i_nominal_gross=1+ss_defaults.i_nominal_net
        # Ensure capital tax rate is reasonable and calculate robust initial guess for Rk_gross
        tau_k_safe = min(max(params.tau_k, 0.0), 0.8)  # Cap at 80% to avoid division by near-zero
        ss_defaults.Rk_gross = (ss_defaults.r_net_real + params.delta) / max(1 - tau_k_safe, 0.2)
        ss_defaults.mc=(params.epsilon-1)/params.epsilon;ss_defaults.Y=1.0
        # Approach: Use target ratios directly and ignore minor inconsistencies for initial guess
        # This creates a reasonable starting point for the solver
        ss_defaults.C = params.cy_ratio * ss_defaults.Y  # 0.6
        ss_defaults.I = params.iy_ratio * ss_defaults.Y  # 0.2  
        # Calculate K from target K/Y ratio instead of I/δ to avoid unrealistic values
        ss_defaults.K = (params.ky_ratio / 4) * ss_defaults.Y  # Use target quarterly K/Y ratio
        ss_defaults.L = params.hours_steady  # 0.33
        
        # Validate production function and adjust Y if needed to be consistent
        Y_from_production = ss_defaults.K**params.alpha * ss_defaults.L**(1-params.alpha)
        if abs(Y_from_production - ss_defaults.Y) > 0.1:  # If major inconsistency
            # Scale Y to match production function, then recalculate ratios
            ss_defaults.Y = Y_from_production
            ss_defaults.C = params.cy_ratio * ss_defaults.Y
            ss_defaults.I = params.iy_ratio * ss_defaults.Y
        ss_defaults.G=params.gy_ratio*ss_defaults.Y 
        # C is already calculated above, but double-check it's consistent with Y = C + I + G
        C_implied = ss_defaults.Y - ss_defaults.I - ss_defaults.G
        if abs(C_implied - ss_defaults.C) > 0.01:
            ss_defaults.C = C_implied  # Use accounting identity 
        if ss_defaults.L > 1e-9 : ss_defaults.w=(1-params.alpha)*ss_defaults.mc*ss_defaults.Y/ss_defaults.L
        else: ss_defaults.w = 2.0
        # Use a reasonable debt level that won't cause negative effective tax rates
        ss_defaults.B_real = max((params.by_ratio/4)*ss_defaults.Y, 0.1*ss_defaults.Y)  # At least 10% of quarterly GDP 
        if ss_defaults.C*(1-params.habit) > 1e-9 : ss_defaults.Lambda=(1-params.beta*params.habit)/(ss_defaults.C*(1-params.habit)*(1+params.tau_c))
        else: ss_defaults.Lambda = 1.0
        ss_defaults.profit=(1-ss_defaults.mc)*ss_defaults.Y; ss_defaults.q=1.0
        ss_defaults.b_star=params.b_star_target_level; A_guess=ss_defaults.C+ss_defaults.I+ss_defaults.G
        Y_star_val_guess=params.ystar_ss; ss_defaults.IM=params.alpha_m*A_guess*ss_defaults.q**(-params.eta_im)
        if params.b_star_target_level==0 and abs(params.beta-1.0)>1e-9: 
            ss_defaults.EX=ss_defaults.IM
            if params.alpha_x>1e-9 and Y_star_val_guess>1e-9 and(params.alpha_m*A_guess)>1e-9 and abs(params.eta_ex+params.eta_im)>1e-9:
                q_rhs=(params.alpha_m*A_guess)/(params.alpha_x*Y_star_val_guess**params.phi_ex)
                if q_rhs > 1e-9: ss_defaults.q=q_rhs**(1/(params.eta_ex+params.eta_im))
                ss_defaults.IM=params.alpha_m*A_guess*ss_defaults.q**(-params.eta_im);ss_defaults.EX=ss_defaults.IM
        else: ss_defaults.EX=params.alpha_x*(Y_star_val_guess**params.phi_ex)*ss_defaults.q**params.eta_ex
        # For tax reforms, test if baseline values work better than tax-adjusted
        if baseline_ss is not None and initial_guess_dict is None:
            # Calculate tax change magnitude to decide on initial guess strategy
            tax_change_magnitude = 0
            if hasattr(self.params, 'tau_c'):
                tax_change_magnitude += abs(self.params.tau_c - 0.10)  # vs baseline 10%
            if hasattr(self.params, 'tau_l'):
                tax_change_magnitude += abs(self.params.tau_l - 0.20)  # vs baseline 20%
            if hasattr(self.params, 'tau_k'):
                tax_change_magnitude += abs(self.params.tau_k - 0.25)  # vs baseline 25%
            if hasattr(self.params, 'tau_f'):
                tax_change_magnitude += abs(self.params.tau_f - 0.30)  # vs baseline 30%
            
            # For small tax changes, baseline values often work better, but avoid the "death valley" around 1.5-2.5pp
            if tax_change_magnitude < 0.015 or (tax_change_magnitude >= 0.015 and tax_change_magnitude <= 0.025):
                # Use tax-adjusted for the problematic 1.5-2.5pp range
                if tax_change_magnitude >= 0.015 and tax_change_magnitude <= 0.025:
                    print(f"Using tax-adjusted initial guess for problematic range (magnitude: {tax_change_magnitude:.3f})")
                    initial_guess_dict = self._compute_tax_adjusted_initial_guess(baseline_ss)
                else:
                    print(f"Using baseline values for small tax change (magnitude: {tax_change_magnitude:.3f})")
                    initial_guess_dict = {var: getattr(baseline_ss, var) 
                                        for var in self.endogenous_vars_solve}
            else:
                print(f"Using tax-adjusted initial guess for large tax change (magnitude: {tax_change_magnitude:.3f})")
                initial_guess_dict = self._compute_tax_adjusted_initial_guess(baseline_ss)
        
        x0_list=[]
        for var_name_solver in self.endogenous_vars_solve:
            val=getattr(ss_defaults,var_name_solver) if initial_guess_dict is None else initial_guess_dict.get(var_name_solver,getattr(ss_defaults,var_name_solver))
            if var_name_solver in self.log_vars_indices:val=np.log(val) if val>1e-9 else np.log(1e-9)
            x0_list.append(val)
        x0=np.array(x0_list)
        # Try different methods with optimized tolerances for tax reforms
        # Detect if this is likely a tax reform (has baseline_ss parameter)
        is_tax_reform = baseline_ss is not None
        
        try:
            if is_tax_reform:
                # For tax reforms, try LM method first as it handles problematic cases better
                opt_result=optimize.root(self.get_equations_for_steady_state,x0,method='lm',options={'xtol':1e-6,'maxiter':3000})
            else:
                # For baseline, use hybr method
                opt_result=optimize.root(self.get_equations_for_steady_state,x0,method='hybr',options={'xtol':1e-6,'maxfev':5000*(len(x0)+1)})
        except:
            try:
                # Fallback: try hybr method
                opt_result=optimize.root(self.get_equations_for_steady_state,x0,method='hybr',options={'xtol':1e-6,'maxfev':5000*(len(x0)+1)})
            except:
                try:
                    # Second fallback: hybr with more iterations
                    opt_result=optimize.root(self.get_equations_for_steady_state,x0,method='hybr',options={'xtol':1e-5,'maxfev':10000*(len(x0)+1)})
                except:
                    try:
                        # Third fallback: LM method
                        opt_result=optimize.root(self.get_equations_for_steady_state,x0,method='lm',options={'xtol':1e-6,'maxiter':3000})
                    except:
                        # Final fallback: Broyden method
                        opt_result=optimize.root(self.get_equations_for_steady_state,x0,method='broyden1',options={'xtol':1e-6,'maxiter':2000})
        # Check if residuals are small even if optimization didn't "succeed"
        if not opt_result.success:
            final_residuals = self.get_equations_for_steady_state(opt_result.x)
            max_residual = np.max(np.abs(final_residuals))
            if max_residual > 0.05:  # Relaxed threshold for tax reforms (was 0.1)
                raise ValueError(f"SS comp failed: {opt_result.message}, max residual: {max_residual:.6e}")
            else:
                print(f"Warning: Optimization didn't converge but residuals are acceptable (max: {max_residual:.6e})")
        ss_values_vec=opt_result.x;final_ss=SteadyState()
        for i,var_name_solver in enumerate(self.endogenous_vars_solve):
            val_to_set=ss_values_vec[i]
            if var_name_solver in self.log_vars_indices:val_to_set=np.exp(val_to_set)
            setattr(final_ss,var_name_solver,val_to_set) 
        final_ss.A_dom=final_ss.C+final_ss.I+final_ss.G;final_ss.Y_star=params.ystar_ss
        final_ss.NX=final_ss.EX-final_ss.IM
        nx_from_bop=final_ss.q*final_ss.b_star*(1-(1/params.beta))
        if abs(final_ss.NX-nx_from_bop)>1e-6:final_ss.NX=nx_from_bop
        # Calculate effective labor tax rate with bounds
        debt_feedback = params.phi_b*((final_ss.B_real/(final_ss.Y if final_ss.Y!=0 else 1e-6))-params.by_ratio/4)
        final_ss.tau_l_effective = max(0.05, min(0.8, params.tau_l_ss + debt_feedback))
        final_ss.Tc=params.tau_c*final_ss.C;final_ss.Tl=final_ss.tau_l_effective*final_ss.w*final_ss.L
        final_ss.Tk=params.tau_k*final_ss.Rk_gross*final_ss.K;final_ss.Tf=params.tau_f*final_ss.profit
        final_ss.T_total_revenue=final_ss.Tc+final_ss.Tl+final_ss.Tk+final_ss.Tf
        final_ss.R_star_gross_real=(1+final_ss.r_net_real);final_ss.i_nominal_net=final_ss.i_nominal_gross-1
        final_ss.A_tfp=1.0
        final_ss.T_transfer=final_ss.T_total_revenue-final_ss.G-final_ss.r_net_real*final_ss.B_real
        self.steady_state=final_ss;return final_ss
    
    def check_steady_state(self, ss: SteadyState) -> Dict[str, float]:
        errors={}; p=self.params
        errors['C/Y']=(ss.C/ss.Y)-p.cy_ratio if ss.Y!=0 else np.nan; errors['I/Y']=(ss.I/ss.Y)-p.iy_ratio if ss.Y!=0 else np.nan
        errors['K/Y_annual']=(ss.K/(4*ss.Y))-p.ky_ratio if ss.Y!=0 else np.nan; errors['Hours']=ss.L-p.hours_steady
        errors['G/Y']=(ss.G/ss.Y)-p.gy_ratio if ss.Y!=0 else np.nan; errors['B/Y_quarterly']=(ss.B_real/ss.Y)-p.by_ratio/4 if ss.Y!=0 else np.nan
        errors['NX/Y']=(ss.NX/ss.Y)-p.nx_y_ratio_target if ss.Y!=0 else np.nan; errors['b_star_level']=ss.b_star-p.b_star_target_level
        x_solve_check=[]
        for var_name_solver in self.endogenous_vars_solve:
            val = getattr(ss, var_name_solver)
            if var_name_solver in self.log_vars_indices: val=np.log(val) if val>1e-9 else np.log(1e-9)
            x_solve_check.append(val)
        equation_residuals=self.get_equations_for_steady_state(np.array(x_solve_check))
        for i,var_name_solver in enumerate(self.endogenous_vars_solve): errors[f'Eq_Resid_{var_name_solver}']=equation_residuals[i]
        return errors

    def get_model_equations(self) -> List[sympy.Eq]:
        if not self.steady_state:
            print("Warning: Steady state must be computed before calling get_model_equations. Computing now...")
            self.compute_steady_state() 
            if not self.steady_state:
                 raise ValueError("Steady state computation failed or not run. Cannot retrieve dynamic equations.")

        ss = self.steady_state 
        p = self.params
        s_ = self._sym 

        # --- Symbolic Variables ---
        # Endogenous variables (names mostly match SteadyState object for clarity)
        C_t, C_tm1, C_tp1 = s_('C', lags=1, leads=1)
        Lambda_t, Lambda_tp1 = s_('Lambda', leads=1) 
        L_t = s_('L') 
        I_t = s_('I')
        K_t, K_tm1 = s_('K', lags=1) 
        B_real_t, B_real_tm1 = s_('B_real', lags=1) 
        q_t, q_tp1 = s_('q', leads=1) 
        b_star_t, b_star_tm1 = s_('b_star', lags=1) 
        Y_t, Y_tm1 = s_('Y', lags=1) 
        A_tfp_t, A_tfp_tm1 = s_('A_tfp', lags=1) 
        W_t = s_('w') 
        Rk_t_gross, Rk_tp1_gross = s_('Rk_gross', leads=1) 
        mc_t = s_('mc') 
        pi_t_gross, pi_tp1_gross = s_('pi_gross', leads=1) 
        profit_t = s_('profit') 
        G_t, G_tm1 = s_('G', lags=1) 
        tau_l_t_effective = s_('tau_l_effective') 
        T_transfer_t = s_('T_transfer') 
        
        i_t_net_nominal, i_tm1_net_nominal = s_('i_nominal_net', lags=1)
        i_t_nominal_gross = s_('i_nominal_gross') # Defined as 1 + i_t_net_nominal

        r_t_net_real, r_tm1_net_real = s_('r_net_real', lags=1) 
        R_star_t_net_real, R_star_tm1_net_real = s_('R_star_net_real', lags=1) 
        
        IM_t = s_('IM'); EX_t = s_('EX'); NX_t = s_('NX')
        A_dom_t = s_('A_dom'); Y_star_t, Y_star_tm1 = s_('Y_star', lags=1)

        eps_a, eps_g, eps_r, eps_ystar = sympy.symbols(' '.join(self.exogenous_shocks_sym_names))
        
        eqs = []
        # --- Household Sector ---
        # Eq 1: Fiscal rule for tau_l_t: tau_l_t = tau_l_ss + phi_b * (B_real_tm1/Y_tm1 - by_target_q)
        # Note: by_ratio is annual debt/GDP, so divide by 4 for quarterly
        eqs.append(sympy.Eq(tau_l_t_effective - (p.tau_l_ss + p.phi_b * (B_real_tm1 / Y_tm1 - p.by_ratio/4)), 0))
        # Eq 2: Budget Constraint (from docs): (1+τc)Cₜ+Iₜ+Bₜ+qₜb\*ₜ = (1-τₗₜ)WₜLₜ+(1-τk)RkₜKₜ₋₁ + R_{t-1}^{gross real}B_{t-1} + q_t R\*_{t-1}^{gross real}b\*_{t-1} + T_transfer_t
        # The R_tm1_net_real and R_star_tm1_net_real are net rates. So (1+net rate) is gross rate.
        eqs.append(sympy.Eq(((1+p.tau_c)*C_t + I_t + B_real_t + q_t*b_star_t -
                        ((1-tau_l_t_effective)*W_t*L_t + (1-p.tau_k)*Rk_t_gross*K_tm1 + 
                         (1+r_tm1_net_real)*B_real_tm1 + q_t*(1+R_star_tm1_net_real)*b_star_tm1 + T_transfer_t)), 0))
        eqs.append(sympy.Eq(1/(C_t-p.habit*C_tm1) - p.beta*p.habit/(C_tp1-p.habit*C_t) - Lambda_t*(1+p.tau_c), 0)) # Eq 3
        eqs.append(sympy.Eq(p.chi*L_t**(1/p.sigma_l) - Lambda_t*(1-tau_l_t_effective)*W_t/(1+p.tau_c), 0)) # Eq 4
        eqs.append(sympy.Eq(Lambda_t - p.beta*Lambda_tp1*(1+r_t_net_real), 0)) # Eq 5
        eqs.append(sympy.Eq(Lambda_t*q_t - p.beta*Lambda_tp1*q_tp1*(1+R_star_t_net_real),0)) # Eq 6
        inv_rate_t = I_t/K_tm1 
        adj_cost = (p.psi/2)*(inv_rate_t - p.delta)**2
        eqs.append(sympy.Eq(K_t - ((1-p.delta)*K_tm1 + I_t*(1-adj_cost)), 0)) # Eq 7
        eqs.append(sympy.Eq(Lambda_t - p.beta*Lambda_tp1*((1-p.tau_k)*Rk_tp1_gross + (1-p.delta)), 0)) # Eq 8

        # --- Firm Sector ---
        eqs.append(sympy.Eq(Y_t - A_tfp_t*K_tm1**p.alpha*L_t**(1-p.alpha), 0)) # Eq 9
        eqs.append(sympy.Eq(log(A_tfp_t/ss.A_tfp) - p.rho_a*log(A_tfp_tm1/ss.A_tfp) - eps_a, 0)) # Eq 10
        eqs.append(sympy.Eq(mc_t*(1-p.alpha)*Y_t/L_t - W_t, 0)) # Eq 11
        eqs.append(sympy.Eq(mc_t*p.alpha*Y_t/K_tm1 - Rk_t_gross, 0)) # Eq 12
        norm_factor = (p.alpha**p.alpha)*((1-p.alpha)**(1-p.alpha))
        eqs.append(sympy.Eq(mc_t*A_tfp_t*norm_factor - W_t**(1-p.alpha)*Rk_t_gross**p.alpha,0)) # Eq 13
        kappa_pi = ((1-p.theta_p)*(1-p.beta*p.theta_p))/p.theta_p
        eqs.append(sympy.Eq(log(pi_t_gross/ss.pi_gross) - p.beta*log(pi_tp1_gross/ss.pi_gross) - kappa_pi*(log(mc_t)-log(ss.mc)),0)) # Eq 14
        eqs.append(sympy.Eq(profit_t - (1-mc_t)*Y_t, 0)) # Eq 15

        # --- Government Sector ---
        total_tax_rev_sym = p.tau_c*C_t + tau_l_t_effective*W_t*L_t + p.tau_k*Rk_t_gross*K_tm1 + p.tau_f*profit_t
        # GBC from docs: B_t = B_{t-1}/pi_t + G_t + T_transfer_t - total_tax_revenue_t
        eqs.append(sympy.Eq(B_real_t - (B_real_tm1/pi_t_gross + G_t + T_transfer_t - total_tax_rev_sym), 0)) # Eq 16
        eqs.append(sympy.Eq(log(G_t/ss.G) - p.rho_g*log(G_tm1/ss.G) - eps_g, 0)) # Eq 17
        eqs.append(sympy.Eq(T_transfer_t - ss.T_transfer, 0)) # Eq 18

        # --- Monetary Policy ---
        eqs.append(sympy.Eq(i_t_net_nominal - (p.rho_r*i_tm1_net_nominal + 
                     (1-p.rho_r)*(ss.i_nominal_net + p.phi_pi*(pi_t_gross - ss.pi_gross) + p.phi_y*(log(Y_t) - log(ss.Y))) 
                     + eps_r), 0)) # Eq 19
        eqs.append(sympy.Eq(i_t_nominal_gross - (1 + i_t_net_nominal), 0)) # Eq 20
        eqs.append(sympy.Eq(i_t_nominal_gross - (1+r_t_net_real)*pi_tp1_gross, 0)) # Eq 21

        # --- International Sector ---
        eqs.append(sympy.Eq(A_dom_t - (C_t + I_t + G_t), 0)) # Eq 22
        eqs.append(sympy.Eq(IM_t - p.alpha_m*A_dom_t*q_t**(-p.eta_im), 0)) # Eq 23
        eqs.append(sympy.Eq(EX_t - p.alpha_x*Y_star_t**p.phi_ex*q_t**p.eta_ex, 0)) # Eq 24
        eqs.append(sympy.Eq(log(Y_star_t/ss.Y_star) - p.rho_ystar*log(Y_star_tm1/ss.Y_star) - eps_ystar, 0)) # Eq 25
        eqs.append(sympy.Eq(NX_t - (EX_t - IM_t), 0)) # Eq 26
        # NFA Accumulation (R_star_tm1_net_real is net real foreign rate on assets from t-1 to t)
        eqs.append(sympy.Eq(q_t*b_star_t - ((1+R_star_tm1_net_real)*q_t*b_star_tm1 + NX_t), 0)) # Eq 27
        eqs.append(sympy.Eq(R_star_t_net_real - r_t_net_real, 0)) # Eq 28 (UIP assumption)

        # --- Market Clearing ---
        eqs.append(sympy.Eq(Y_t - (C_t + I_t + G_t + NX_t), 0)) # Eq 29
        
        # Verify all symbols are defined:
        # C_t,C_tm1,C_tp1,Lambda_t,Lambda_tp1,L_t,I_t,K_t,K_tm1,B_real_t,B_real_tm1,q_t,q_tp1,b_star_t,b_star_tm1,
        # Y_t,Y_tm1,A_tfp_t,A_tfp_tm1,W_t,Rk_t_gross,Rk_tp1_gross,mc_t,pi_t_gross,pi_tp1_gross,profit_t,
        # G_t,G_tm1,tau_l_t_effective,T_transfer_t,i_t_net_nominal,i_tm1_net_nominal,i_t_nominal_gross,
        # r_t_net_real,r_tm1_net_real,R_star_t_net_real,R_star_tm1_net_real,IM_t,EX_t,NX_t,A_dom_t,Y_star_t,Y_star_tm1,
        # eps_a,eps_g,eps_r,eps_ystar
        # All seem to be defined and used correctly.

        return eqs

def load_model(config_path: str) -> DSGEModel:
    params = ModelParameters.from_json(config_path)
    return DSGEModel(params)
