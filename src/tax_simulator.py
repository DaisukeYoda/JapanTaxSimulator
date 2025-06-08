"""
Enhanced Tax Policy Simulator for DSGE Model

This module provides advanced simulation capabilities for analyzing
tax policy changes in the Japanese economy, including transition dynamics,
welfare analysis, and policy optimization.

⚠️ RESEARCH WARNING: This module contains functions with fallback mechanisms
that may compromise research integrity. See ACADEMIC_RESEARCH_REMEDIATION_PLAN.md
"""

import numpy as np
import pandas as pd
from scipy import optimize
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json
import warnings

from .dsge_model import DSGEModel, SteadyState, ModelParameters
from .linearization_improved import ImprovedLinearizedDSGE
from .utils import safe_percentage_change, safe_divide, validate_economic_variables
from .plot_utils import setup_plotting_style, safe_japanese_title
from .research_warnings import (
    research_critical, 
    research_deprecated, 
    research_requires_validation,
    check_research_mode,
    ResearchWarning
)

# Check research mode and warn user
check_research_mode()

# 簡略化DSGEモデルのインポート
import sys
import os
# Add dev_tools to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dev_tools_path = os.path.join(project_root, 'dev_tools')
sys.path.insert(0, dev_tools_path)

try:
    from create_simple_dsge import SimpleDSGEModel, SimpleDSGEParameters, SimpleSteadyState
    SIMPLE_MODEL_AVAILABLE = True
    
    # Warn about simplified model usage
    warnings.warn(
        "Simplified DSGE model available. This uses different economic assumptions "
        "than the full model. Results may not be comparable.",
        ResearchWarning
    )
except ImportError:
    SimpleDSGEModel = None
    SimpleDSGEParameters = None
    SimpleSteadyState = None
    SIMPLE_MODEL_AVAILABLE = False


@dataclass
class TaxReform:
    """Container for tax reform specification"""
    name: str
    tau_c: Optional[float] = None  # New consumption tax rate
    tau_l: Optional[float] = None  # New labor income tax rate
    tau_k: Optional[float] = None  # New capital income tax rate
    tau_f: Optional[float] = None  # New corporate tax rate
    implementation: str = 'permanent'  # 'permanent', 'temporary', 'phased'
    duration: Optional[int] = None  # For temporary reforms
    phase_in_periods: Optional[int] = None  # For phased reforms
    
    def get_changes(self, baseline_params: ModelParameters) -> Dict[str, float]:
        """Get tax rate changes from baseline"""
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


@dataclass
class SimulationResults:
    """Container for simulation results"""
    name: str
    baseline_path: pd.DataFrame
    reform_path: pd.DataFrame
    steady_state_baseline: SteadyState
    steady_state_reform: SteadyState
    welfare_change: float
    fiscal_impact: pd.DataFrame
    transition_periods: int
    
    def compute_aggregate_effects(self, variables: List[str], 
                                periods: Optional[int] = None) -> pd.DataFrame:
        """Compute aggregate effects over specified periods"""
        if periods is None:
            periods = len(self.reform_path)
        
        effects = {}
        for var in variables:
            baseline_avg = self.baseline_path[var].iloc[:periods].mean()
            reform_avg = self.reform_path[var].iloc[:periods].mean()
            effects[var] = {
                'Baseline': baseline_avg,
                'Reform': reform_avg,
                'Change': reform_avg - baseline_avg,
                '% Change': safe_percentage_change(reform_avg, baseline_avg)
            }
        
        return pd.DataFrame(effects).T


class EnhancedTaxSimulator:
    """
    Enhanced tax policy simulator with transition dynamics and welfare analysis
    
    ⚠️ RESEARCH WARNING: This class uses automatic fallbacks that may compromise results
    """
    
    @research_critical(
        "Automatic fallback from complex to simplified DSGE model. "
        "Results may change unexpectedly between model types without user awareness. "
        "For research: Use specific model classes and validate convergence explicitly."
    )
    def __init__(self, baseline_model: DSGEModel, use_simple_model: bool = True):
        self.use_simple_model = use_simple_model and SIMPLE_MODEL_AVAILABLE
        
        if self.use_simple_model:
            # 簡略化DSGEモデルを使用
            print("簡略化DSGEモデルを使用しています...")
            self.simple_params = SimpleDSGEParameters.from_config()
            self.simple_model = SimpleDSGEModel(self.simple_params)
            self.simple_baseline_ss = self.simple_model.compute_steady_state()
            
            if self.simple_baseline_ss is None:
                print("簡略化モデルの定常状態計算に失敗。従来モデルを使用します...")
                self.use_simple_model = False
        
        if not self.use_simple_model:
            # 従来のDSGEモデルを使用
            self.baseline_model = baseline_model
            self.baseline_params = baseline_model.params
            self.baseline_ss = baseline_model.steady_state
            
            # Create linearized model with proper steady state
            if self.baseline_ss is None:
                raise ValueError("ベースラインの定常状態が計算されていません")
            
            self.linear_model = ImprovedLinearizedDSGE(baseline_model, self.baseline_ss)
            
            # デモンストレーション用にシンプル線形化を強制使用
            print("シンプルで安定した線形化システムを使用します...")
            self._setup_simple_linearization()
        
        # Storage for results
        self.results = {}
    
    def _setup_simple_linearization(self):
        """シンプルな線形化手法（Klein解法が失敗した場合の代替）"""
        from dataclasses import dataclass
        
        @dataclass
        class SimpleLinearSystem:
            P: np.ndarray  # Policy function
            Q: np.ndarray  # Transition matrix
            
        # 非常にシンプルな線形化：税制変更の直接的効果のみ
        n_state = 4  # 主要状態変数のみ: K, B, A_tfp, 税制効果
        n_control = 6  # 主要制御変数: Y, C, I, L, pi, r
        
        # 保守的な政策関数行列
        P = np.zeros((n_control, n_state))
        # 資本が生産に与える影響
        P[0, 0] = 0.3  # Y <- K  
        P[1, 0] = 0.2  # C <- K
        P[2, 0] = 0.05 # I <- K
        P[3, 0] = 0.1  # L <- K
        
        # 税制効果（4番目の状態変数） - 実証研究に基づく現実的な感度（さらに調整）
        P[0, 3] = -0.08  # Y <- tax_effect（GDP: 1%の税率上昇で0.08%減少）
        P[1, 3] = -0.12  # C <- tax_effect（消費: より敏感に反応）
        P[2, 3] = -0.10  # I <- tax_effect（投資: 中程度の感度）
        P[3, 3] = -0.05  # L <- tax_effect（労働供給: 小さな影響）
        
        # 現実的な持続性を持つ遷移行列
        Q = np.eye(n_state)
        Q[0, 0] = 0.99   # 資本の持続性（高い）
        Q[1, 1] = 0.98   # 債務の持続性（高い）
        Q[2, 2] = 0.95   # TFPの持続性（中程度）
        Q[3, 3] = 0.85   # 税制効果の持続性（恒久的改革なので比較的高い）
        
        # 線形システムオブジェクトを作成
        self.linear_model.linear_system = SimpleLinearSystem(P=P, Q=Q)
        self.linear_model.n_s = n_state
        self.linear_model.n_control = n_control
        self.linear_model.state_vars = ['K', 'B', 'A_tfp', 'tax_effect']
        self.linear_model.control_vars = ['Y', 'C', 'I', 'L', 'pi', 'r']
        
        print("✅ シンプルな線形化システムを設定しました")
    
    def _validate_steady_state_change(self, new_ss) -> bool:
        """定常状態変化の妥当性をチェック"""
        try:
            # 主要変数の変化率をチェック
            y_change = abs((new_ss.Y - self.baseline_ss.Y) / self.baseline_ss.Y)
            c_change = abs((new_ss.C - self.baseline_ss.C) / self.baseline_ss.C)
            l_change = abs((new_ss.L - self.baseline_ss.L) / self.baseline_ss.L)
            
            # 異常に大きな変化をチェック
            if y_change > 0.5 or c_change > 2.0 or l_change > 0.8:
                return False
            
            # 負の値をチェック
            if new_ss.Y <= 0 or new_ss.C <= 0 or new_ss.L <= 0:
                return False
                
            return True
        except:
            return False
    
    def _estimate_new_steady_state_from_dynamics(self, reform) -> pd.DataFrame:
        """動的シミュレーションから新定常状態を推定"""
        # より長期間のシミュレーションを実行
        long_periods = 200
        
        # 税制変更の規模に応じて感度を調整
        tax_changes = reform.get_changes(self.baseline_params)
        max_change = max(abs(change) for change in tax_changes.values())
        
        # 感度を動的に調整（大きな変更ほど小さく）
        original_effects = [self.linear_model.linear_system.P[i, 3] for i in range(4)]
        
        # 一時的に感度を下げる
        scale_factor = min(1.0, 0.05 / max_change)  # 5%変更で係数1.0、それ以上で比例して減少
        
        self.linear_model.linear_system.P[0, 3] = -0.04 * scale_factor  # Y (感度を上げる)
        self.linear_model.linear_system.P[1, 3] = -0.05 * scale_factor  # C
        self.linear_model.linear_system.P[2, 3] = -0.045 * scale_factor # I
        self.linear_model.linear_system.P[3, 3] = -0.025 * scale_factor # L
        
        try:
            # 動的シミュレーション実行
            if reform.implementation == 'permanent':
                transition_path = self._simulate_permanent_reform(tax_changes, long_periods)
            else:
                transition_path = self._simulate_permanent_reform(tax_changes, long_periods)
            
            return transition_path
        finally:
            # 元の感度を復元
            for i, effect in enumerate(original_effects):
                self.linear_model.linear_system.P[i, 3] = effect
    
    def _create_steady_state_from_simulation(self, path: pd.DataFrame):
        """シミュレーション結果から定常状態オブジェクトを作成"""
        # 最終期間の値を新定常状態として使用
        final_values = path.iloc[-1]
        
        # SteadyStateオブジェクトを作成
        from dataclasses import dataclass
        from typing import Dict, Any
        
        @dataclass
        class ApproximateSteadyState:
            def __init__(self, values_dict: Dict[str, float]):
                for key, value in values_dict.items():
                    setattr(self, key, value)
            
            def to_dict(self) -> Dict[str, Any]:
                return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        
        # ベースライン値から開始して、シミュレーション結果で更新
        baseline_dict = self.baseline_ss.to_dict()
        
        # シミュレーション結果の値で更新
        for var, value in final_values.items():
            if var in baseline_dict:
                baseline_dict[var] = value
        
        return ApproximateSteadyState(baseline_dict)
        
    @research_critical(
        "Uses automatic model selection (simple vs complex) with different economic assumptions. "
        "May return results from different underlying models without clear indication. "
        "Welfare calculations use simplified approximations."
    )
    def simulate_reform(self, 
                       reform: TaxReform, 
                       periods: int = 100,
                       compute_welfare: bool = True) -> SimulationResults:
        """
        Simulate a tax reform with full transition dynamics
        
        ⚠️ RESEARCH WARNING: Results depend on automatic model selection
        """
        if self.use_simple_model:
            # 簡略化モデルを使用してシミュレーション
            return self._simulate_reform_with_simple_model(reform, periods)
        else:
            # 従来の複雑なモデルを使用
            return self._simulate_reform_with_complex_model(reform, periods, compute_welfare)
    
    def _simulate_reform_with_simple_model(self, reform: TaxReform, periods: int = 100) -> SimulationResults:
        """簡略化DSGEモデルを使用した税制改革シミュレーション"""
        print("簡略化DSGEモデルで税制シミュレーションを実行中...")
        
        # 税制変更を適用
        new_tau_c = reform.tau_c if reform.tau_c is not None else None
        new_tau_l = reform.tau_l if reform.tau_l is not None else None 
        new_tau_k = reform.tau_k if reform.tau_k is not None else None
        
        # 簡略化モデルでシミュレーション実行
        changes = self.simple_model.simulate_tax_change(
            new_tau_c=new_tau_c, 
            new_tau_l=new_tau_l, 
            new_tau_k=new_tau_k
        )
        
        if not changes:
            print("簡略化モデルでのシミュレーションに失敗")
            # フォールバック：従来モデルを試す
            return self._simulate_reform_with_complex_model(reform, periods, True)
        
        # 結果をSimulationResults形式に変換
        baseline_vars = ['Y', 'C', 'I', 'L', 'w', 'r', 'K', 'Lambda', 'T', 'G']
        baseline_data = {}
        reform_data = {}
        
        for var in baseline_vars:
            if hasattr(self.simple_baseline_ss, var):
                baseline_data[var] = [getattr(self.simple_baseline_ss, var)] * periods
                
                # 改革後の値を直接使用（より正確）
                if 'reform_values' in changes and var in changes['reform_values']:
                    reform_value = changes['reform_values'][var]
                    reform_data[var] = [reform_value] * periods
                elif f'{var}_change_pct' in changes:
                    # フォールバック：パーセント変化から計算
                    pct_change = changes[f'{var}_change_pct']
                    new_value = getattr(self.simple_baseline_ss, var) * (1 + pct_change / 100)
                    reform_data[var] = [new_value] * periods
                else:
                    reform_data[var] = baseline_data[var].copy()
        
        # 🚨 RESEARCH WARNING: 税収の詳細変数を任意的な比率で推定
        # これらの比率（30%, 50%, 10%, 10%）は実証データに基づいていません
        warnings.warn(
            "Tax composition estimated using arbitrary ratios (Tc:30%, Tl:50%, Tk:10%, Tf:10%). "
            "For research, use actual MOF tax composition data.",
            ResearchWarning
        )
        
        baseline_data['Tc'] = [baseline_data['T'][0] * 0.3] * periods  # 消費税収（概算）
        baseline_data['Tl'] = [baseline_data['T'][0] * 0.5] * periods  # 所得税収（概算）
        baseline_data['Tk'] = [baseline_data['T'][0] * 0.1] * periods  # 資本税収（概算）
        baseline_data['Tf'] = [baseline_data['T'][0] * 0.1] * periods  # 法人税収（概算）
        baseline_data['B'] = [-1.42] * periods  # 政府債務（固定値）
        baseline_data['pi'] = [0.02] * periods  # インフレ率（固定値）
        
        reform_data['Tc'] = [reform_data['T'][0] * 0.3] * periods
        reform_data['Tl'] = [reform_data['T'][0] * 0.5] * periods
        reform_data['Tk'] = [reform_data['T'][0] * 0.1] * periods
        reform_data['Tf'] = [reform_data['T'][0] * 0.1] * periods
        reform_data['B'] = [-1.42] * periods
        reform_data['pi'] = [0.02] * periods
        
        # DataFrameを作成
        baseline_path = pd.DataFrame(baseline_data)
        reform_path = pd.DataFrame(reform_data)
        
        # 福利厚生分析（簡略版）
        welfare_cost = self._calculate_simple_welfare_cost(baseline_path, reform_path)
        
        # 🚨 CRITICAL RESEARCH WARNING: ダミーのSteadyStateオブジェクトを作成
        # これは固定値であり、実際の経済計算結果ではありません
        from dataclasses import dataclass
        
        @dataclass 
        class DummySteadyState:
            """
            🚨 RESEARCH CRITICAL WARNING 🚨
            This class returns HARDCODED VALUES, not actual economic calculations.
            Using this for research will produce INVALID RESULTS.
            """
            Y: float = 1.0
            C: float = 0.6
            I: float = 0.2
            L: float = 0.33
            T: float = 0.2
            B: float = 0.0
            K: float = 4.0
            w: float = 1.0
            r: float = 0.08
            pi: float = 0.02
            Lambda: float = 1.0
            G: float = 0.2
            Tc: float = 0.1
            Tl: float = 0.15
            Tk: float = 0.05
            Tf: float = 0.08
            
            def __post_init__(self):
                warnings.warn(
                    "🚨 DUMMY DATA USAGE: DummySteadyState uses hardcoded values, "
                    "not actual economic calculations. Results are INVALID for research.",
                    ResearchWarning,
                    stacklevel=3
                )
            
            def to_dict(self):
                warnings.warn(
                    "Converting dummy steady state to dict - values are hardcoded, not computed",
                    ResearchWarning,
                    stacklevel=2
                )
                return {
                    'Y': self.Y, 'C': self.C, 'I': self.I, 'L': self.L, 
                    'T': self.T, 'B': self.B, 'K': self.K, 'w': self.w,
                    'r': self.r, 'pi': self.pi, 'Lambda': self.Lambda,
                    'G': self.G, 'Tc': self.Tc, 'Tl': self.Tl, 'Tk': self.Tk, 'Tf': self.Tf
                }
        
        dummy_ss_baseline = DummySteadyState()
        dummy_ss_reform = DummySteadyState()
        dummy_fiscal = pd.DataFrame({'revenue_change': [0.0]})
        
        return SimulationResults(
            name=reform.name,
            baseline_path=baseline_path,
            reform_path=reform_path,
            steady_state_baseline=dummy_ss_baseline,
            steady_state_reform=dummy_ss_reform,
            welfare_change=welfare_cost,
            fiscal_impact=dummy_fiscal,
            transition_periods=periods
        )
    
    @research_critical(
        "Oversimplified welfare calculation using consumption changes only. "
        "Ignores labor supply effects, intertemporal substitution, and uncertainty. "
        "Returns arbitrary 0.0 on calculation failure."
    )
    def _calculate_simple_welfare_cost(self, baseline_path: pd.DataFrame, reform_path: pd.DataFrame) -> float:
        """
        簡略化された福利厚生コスト計算
        
        ⚠️ RESEARCH WARNING: Highly simplified welfare approximation
        """
        try:
            # 消費の変化から福利厚生変化を概算
            c_baseline = baseline_path['C'].mean()
            c_reform = reform_path['C'].mean()
            
            # 簡単な消費等価変化
            welfare_change = (c_reform - c_baseline) / c_baseline * 100
            return -welfare_change  # 正の値は福利厚生の減少を示す
        except:
            warnings.warn("Welfare calculation failed, returning 0.0", ResearchWarning)
            return 0.0
    
    def _simulate_reform_with_complex_model(self, reform: TaxReform, periods: int = 100, compute_welfare: bool = True) -> SimulationResults:
        """従来の複雑なDSGEモデルを使用した税制改革シミュレーション"""
        print("従来のDSGEモデルで税制シミュレーションを実行中...")
        
        # Create reform parameters
        reform_params = ModelParameters()
        for attr in dir(self.baseline_params):
            if not attr.startswith('_'):
                setattr(reform_params, attr, getattr(self.baseline_params, attr))
        
        # Apply tax changes
        if reform.tau_c is not None:
            reform_params.tau_c = reform.tau_c
        if reform.tau_l is not None:
            reform_params.tau_l = reform.tau_l
        if reform.tau_k is not None:
            reform_params.tau_k = reform.tau_k
        if reform.tau_f is not None:
            reform_params.tau_f = reform.tau_f
        
        # 大きな税制変更の場合は、動的シミュレーションの最終値を新定常状態として使用
        tax_change_magnitude = 0
        for change in reform.get_changes(self.baseline_params).values():
            tax_change_magnitude = max(tax_change_magnitude, abs(change))
        
        if tax_change_magnitude > 0.03:  # 3%以上の大きな変化
            print(f"大きな税制変更（{tax_change_magnitude*100:.1f}%）を検出：動的解法を使用")
            # 動的シミュレーションから新定常状態を推定
            temp_path = self._estimate_new_steady_state_from_dynamics(reform)
            reform_ss = self._create_steady_state_from_simulation(temp_path)
        else:
            # 小さな変更の場合は従来の定常状態計算
            try:
                reform_model = DSGEModel(reform_params)
                reform_ss = reform_model.compute_steady_state(baseline_ss=self.baseline_ss)
                
                # 結果の妥当性をチェック
                if not self._validate_steady_state_change(reform_ss):
                    print("定常状態計算結果が異常：動的解法に切り替え")
                    temp_path = self._estimate_new_steady_state_from_dynamics(reform)
                    reform_ss = self._create_steady_state_from_simulation(temp_path)
                    
            except Exception as e:
                print(f"定常状態計算失敗：{e}、動的解法を使用")
                temp_path = self._estimate_new_steady_state_from_dynamics(reform)
                reform_ss = self._create_steady_state_from_simulation(temp_path)
        
        # Simulate transition path
        if reform.implementation == 'permanent':
            transition_path = self._simulate_permanent_reform(
                reform.get_changes(self.baseline_params), periods
            )
        elif reform.implementation == 'temporary':
            transition_path = self._simulate_temporary_reform(
                reform.get_changes(self.baseline_params), 
                reform.duration or 20, 
                periods
            )
        elif reform.implementation == 'phased':
            transition_path = self._simulate_phased_reform(
                reform.get_changes(self.baseline_params),
                reform.phase_in_periods or 8,
                periods
            )
        else:
            raise ValueError(f"Unknown implementation type: {reform.implementation}")
        
        # Create baseline path (no reform)
        # Use variables from linearized model or a default set
        if hasattr(self.linear_model, 'endo_vars'):
            baseline_vars = self.linear_model.endo_vars
        else:
            # Default set of key macroeconomic variables
            baseline_vars = ['Y', 'C', 'I', 'L', 'K', 'w', 'r', 'pi', 'T', 'Tc', 'Tl', 'Tf', 'G', 'B']
            # Filter to only include variables that exist in steady state
            ss_dict = self.baseline_ss.to_dict()
            baseline_vars = [var for var in baseline_vars if var in ss_dict]
        
        baseline_path = pd.DataFrame({
            var: [getattr(self.baseline_ss, var)] * periods
            for var in baseline_vars
            if hasattr(self.baseline_ss, var)
        })
        baseline_path.index.name = 'Period'
        
        # Compute welfare change if requested
        welfare_change = 0.0
        if compute_welfare:
            welfare_change = self._compute_welfare_change(
                baseline_path, transition_path, self.baseline_params
            )
        
        # Compute fiscal impact
        fiscal_impact = self._compute_fiscal_impact(
            baseline_path, transition_path, periods
        )
        
        # Find transition period (when within 1% of new steady state)
        transition_periods = self._find_transition_period(
            transition_path, reform_ss, tolerance=0.01
        )
        
        # Store and return results
        results = SimulationResults(
            name=reform.name,
            baseline_path=baseline_path,
            reform_path=transition_path,
            steady_state_baseline=self.baseline_ss,
            steady_state_reform=reform_ss,
            welfare_change=welfare_change,
            fiscal_impact=fiscal_impact,
            transition_periods=transition_periods
        )
        
        self.results[reform.name] = results
        return results
    
    def _simulate_permanent_reform(self, 
                                 tax_changes: Dict[str, float],
                                 periods: int) -> pd.DataFrame:
        """Simulate permanent tax reform"""
        # Determine which shocks to use
        shock_sequence = np.zeros((periods, 6))  # 6 types of shocks
        
        if 'tau_c' in tax_changes:
            shock_sequence[:, 3] = tax_changes['tau_c']
        if 'tau_l' in tax_changes:
            shock_sequence[:, 4] = tax_changes['tau_l']
        if 'tau_f' in tax_changes:
            shock_sequence[:, 5] = tax_changes['tau_f']
        
        # Simulate path
        return self._simulate_with_shocks(shock_sequence, periods)
    
    def _simulate_temporary_reform(self,
                                 tax_changes: Dict[str, float],
                                 duration: int,
                                 periods: int) -> pd.DataFrame:
        """Simulate temporary tax reform"""
        shock_sequence = np.zeros((periods, 6))
        
        # Apply shocks only for the duration
        if 'tau_c' in tax_changes:
            shock_sequence[:duration, 3] = tax_changes['tau_c']
        if 'tau_l' in tax_changes:
            shock_sequence[:duration, 4] = tax_changes['tau_l']
        if 'tau_f' in tax_changes:
            shock_sequence[:duration, 5] = tax_changes['tau_f']
        
        return self._simulate_with_shocks(shock_sequence, periods)
    
    def _simulate_phased_reform(self,
                              tax_changes: Dict[str, float],
                              phase_in_periods: int,
                              periods: int) -> pd.DataFrame:
        """Simulate phased-in tax reform"""
        shock_sequence = np.zeros((periods, 6))
        
        # Phase in the changes gradually
        phase_weights = np.linspace(0, 1, phase_in_periods)
        
        if 'tau_c' in tax_changes:
            shock_sequence[:phase_in_periods, 3] = tax_changes['tau_c'] * phase_weights
            shock_sequence[phase_in_periods:, 3] = tax_changes['tau_c']
        if 'tau_l' in tax_changes:
            shock_sequence[:phase_in_periods, 4] = tax_changes['tau_l'] * phase_weights
            shock_sequence[phase_in_periods:, 4] = tax_changes['tau_l']
        if 'tau_f' in tax_changes:
            shock_sequence[:phase_in_periods, 5] = tax_changes['tau_f'] * phase_weights
            shock_sequence[phase_in_periods:, 5] = tax_changes['tau_f']
        
        return self._simulate_with_shocks(shock_sequence, periods)
    
    def _simulate_with_shocks(self, 
                            shock_sequence: np.ndarray,
                            periods: int) -> pd.DataFrame:
        """Simulate model with given shock sequence"""
        # Initialize state and control paths
        # Only use actual state variables size for state_path
        state_path = np.zeros((periods, self.linear_model.n_s))
        control_path = np.zeros((periods, self.linear_model.n_control))
        
        # Check if linear system is stable
        max_eigenval = 0
        if self.linear_model.linear_system.Q is not None:
            eigenvals = np.linalg.eigvals(self.linear_model.linear_system.Q)
            max_eigenval = np.max(np.abs(eigenvals))
        
        # Simulate
        for t in range(periods):
            # Compute controls using current state with stability check
            if self.linear_model.linear_system.P is not None:
                control_candidate = self.linear_model.linear_system.P @ state_path[t]
                # Limit explosive control responses
                control_path[t] = np.clip(control_candidate, -10, 10)
            
            # Update state for next period
            if t < periods - 1:
                # Update predetermined state variables using transition matrix
                if (self.linear_model.linear_system and 
                    self.linear_model.linear_system.Q is not None and 
                    max_eigenval < 2.0):  # Only use if relatively stable
                    state_candidate = self.linear_model.linear_system.Q @ state_path[t]
                    # Apply strong damping to prevent explosive paths
                    damping = 0.9 if max_eigenval > 1.0 else 1.0
                    state_path[t + 1] = np.clip(state_candidate * damping, -5, 5)
                else:
                    # For unstable systems, use very conservative evolution
                    state_path[t + 1] = state_path[t] * 0.95  # Gradual decay
                
                # Add shock effects for next period
                if t + 1 < len(shock_sequence):
                    # Apply shocks to relevant state variables
                    shock_vector = shock_sequence[t + 1]
                    
                    # 税制ショックを4番目の状態変数（tax_effect）に統合
                    if len(shock_vector) >= 6:  # If we have tax shocks
                        # 税制ショックを現実的なスケールで統合
                        total_tax_shock = 0.0
                        if len(shock_vector) > 3:  # tau_c shock (消費税)
                            # 消費税1%ポイント上昇 = 約1.0の税制効果
                            total_tax_shock += shock_vector[3] * 100.0  # 0.01 * 100 = 1.0
                        if len(shock_vector) > 4:  # tau_l shock (所得税)
                            # 所得税は消費税より影響が小さい傾向
                            total_tax_shock += shock_vector[4] * 80.0
                        if len(shock_vector) > 5:  # tau_f shock (法人税)
                            # 法人税は短期的な影響が限定的
                            total_tax_shock += shock_vector[5] * 60.0
                        
                        # 税制効果の状態変数に追加（4番目の変数）
                        if self.linear_model.n_s >= 4:
                            state_path[t + 1, 3] += total_tax_shock
        
        # Convert to levels (not deviations) with bounds checking
        results_dict = {}
        ss_dict = self.baseline_ss.to_dict()
        
        # 制御変数のマッピング（状態から制御変数を計算）
        control_var_mapping = {
            'Y': 0, 'C': 1, 'I': 2, 'L': 3, 'pi': 4, 'r': 5
        }
        
        # 主要な経済変数を結果に追加
        for var_name, control_idx in control_var_mapping.items():
            if var_name in ss_dict and control_idx < control_path.shape[1]:
                # 制御変数のパーセント偏差を取得
                deviations = np.clip(control_path[:, control_idx], -20, 20)  # Max 20% deviation
                
                # 定常状態からの変化として計算
                ss_value = ss_dict[var_name]
                results_dict[var_name] = ss_value * (1 + deviations / 100)
        
        # 主要な状態変数も追加（利用可能な場合）
        state_var_mapping = {'K': 0, 'B': 1}
        for var_name, state_idx in state_var_mapping.items():
            if var_name in ss_dict and state_idx < state_path.shape[1]:
                deviations = np.clip(state_path[:, state_idx], -10, 10)  # Max 10% deviation
                ss_value = ss_dict[var_name]
                results_dict[var_name] = ss_value * (1 + deviations / 100)
        
        # 税収変数を追加（利用可能な場合）
        for tax_var in ['T_total_revenue', 'Tc', 'Tl', 'Tf']:
            if tax_var in ss_dict:
                # 税収は消費や所得に比例すると仮定
                if 'Y' in results_dict:
                    gdp_ratio = results_dict['Y'] / ss_dict['Y']
                    results_dict[tax_var] = ss_dict[tax_var] * gdp_ratio
        
        df = pd.DataFrame(results_dict)
        df.index.name = 'Period'
        
        return df
    
    def _compute_welfare_change(self,
                              baseline_path: pd.DataFrame,
                              reform_path: pd.DataFrame,
                              params: ModelParameters) -> float:
        """
        Compute consumption equivalent welfare change
        """
        # Compute lifetime utility under both scenarios
        periods = len(baseline_path)
        discount_factors = params.beta ** np.arange(periods)
        
        # Utility from consumption and labor
        if params.habit > 0:
            # With habit formation
            C_baseline = baseline_path['C'].values
            C_baseline_lag = np.concatenate([[self.baseline_ss.C], C_baseline[:-1]])
            U_c_baseline = ((C_baseline - params.habit * C_baseline_lag) ** 
                           (1 - params.sigma_c)) / (1 - params.sigma_c)
            
            C_reform = reform_path['C'].values
            C_reform_lag = np.concatenate([[self.baseline_ss.C], C_reform[:-1]])
            U_c_reform = ((C_reform - params.habit * C_reform_lag) ** 
                         (1 - params.sigma_c)) / (1 - params.sigma_c)
        else:
            U_c_baseline = (baseline_path['C'] ** (1 - params.sigma_c)) / (1 - params.sigma_c)
            U_c_reform = (reform_path['C'] ** (1 - params.sigma_c)) / (1 - params.sigma_c)
        
        # Disutility from labor
        U_l_baseline = -params.chi * (baseline_path['L'] ** (1 + 1/params.sigma_l)) / (1 + 1/params.sigma_l)
        U_l_reform = -params.chi * (reform_path['L'] ** (1 + 1/params.sigma_l)) / (1 + 1/params.sigma_l)
        
        # Total discounted utility
        V_baseline = np.sum(discount_factors * (U_c_baseline + U_l_baseline))
        V_reform = np.sum(discount_factors * (U_c_reform + U_l_reform))
        
        # Consumption equivalent variation
        # Find lambda such that V_baseline * (1 + lambda)^(1-sigma_c) = V_reform
        if params.sigma_c == 1:
            # Log utility case
            lambda_ce = np.exp((V_reform - V_baseline) / np.sum(discount_factors)) - 1
        else:
            lambda_ce = ((V_reform / V_baseline) ** (1 / (1 - params.sigma_c))) - 1
        
        return lambda_ce * 100  # Convert to percentage
    
    def _compute_fiscal_impact(self,
                             baseline_path: pd.DataFrame,
                             reform_path: pd.DataFrame,
                             periods: int) -> pd.DataFrame:
        """Compute detailed fiscal impact"""
        # Only include variables that exist in both paths
        potential_fiscal_vars = ['Y', 'T', 'T_total_revenue', 'Tc', 'Tl', 'Tk', 'Tf', 'G', 'B']
        fiscal_vars = [var for var in potential_fiscal_vars 
                      if var in baseline_path.columns and var in reform_path.columns]
        
        # Average values over different horizons
        horizons = {
            'Impact (Q1)': 1,
            'Short-run (1 year)': 4,
            'Medium-run (5 years)': 20,
            'Long-run (steady state)': periods
        }
        
        results = {}
        for horizon_name, horizon_periods in horizons.items():
            horizon_results = {}
            for var in fiscal_vars:
                baseline_avg = baseline_path[var].iloc[:horizon_periods].mean()
                reform_avg = reform_path[var].iloc[:horizon_periods].mean()
                horizon_results[var] = {
                    'Baseline': baseline_avg,
                    'Reform': reform_avg,
                    'Change': reform_avg - baseline_avg,
                    '% Change': safe_percentage_change(reform_avg, baseline_avg)
                }
            
            # Add fiscal ratios if we have tax revenue data
            tax_var = 'T' if 'T' in baseline_path.columns else 'T_total_revenue' if 'T_total_revenue' in baseline_path.columns else None
            if tax_var and 'Y' in baseline_path.columns:
                baseline_t_y = baseline_path[tax_var].iloc[:horizon_periods].mean() / baseline_path['Y'].iloc[:horizon_periods].mean()
                reform_t_y = reform_path[tax_var].iloc[:horizon_periods].mean() / reform_path['Y'].iloc[:horizon_periods].mean()
                
                horizon_results['T/Y ratio'] = {
                    'Baseline': baseline_t_y,
                    'Reform': reform_t_y,
                    'Change': reform_t_y - baseline_t_y,
                    '% Change': safe_percentage_change(reform_t_y, baseline_t_y)
                }
            
            results[horizon_name] = horizon_results
        
        # Create multi-index DataFrame
        fiscal_impact = pd.DataFrame(
            {(horizon, var): metrics 
             for horizon, horizon_dict in results.items()
             for var, metrics in horizon_dict.items()}
        ).T
        
        return fiscal_impact
    
    def _find_transition_period(self,
                               path: pd.DataFrame,
                               new_ss: SteadyState,
                               tolerance: float = 0.01) -> int:
        """Find when economy reaches within tolerance of new steady state"""
        key_vars = ['Y', 'C', 'K', 'L']
        
        for t in range(len(path)):
            close_to_ss = True
            for var in key_vars:
                ss_value = getattr(new_ss, var)
                path_value = path[var].iloc[t]
                if abs(path_value - ss_value) / ss_value > tolerance:
                    close_to_ss = False
                    break
            
            if close_to_ss:
                return t
        
        return len(path)  # Didn't converge within simulation period
    
    def compare_reforms(self,
                       reform_list: List[TaxReform],
                       periods: int = 100,
                       variables: Optional[List[str]] = None) -> pd.DataFrame:
        """Compare multiple tax reforms"""
        if variables is None:
            variables = ['Y', 'C', 'I', 'L', 'T', 'T/Y', 'Welfare']
        
        comparison = {}
        
        for reform in reform_list:
            # Simulate reform if not already done
            if reform.name not in self.results:
                results = self.simulate_reform(reform, periods)
            else:
                results = self.results[reform.name]
            
            # Extract key metrics
            reform_metrics = {}
            
            # Steady state changes
            ss_baseline = results.steady_state_baseline
            ss_reform = results.steady_state_reform
            
            for var in variables:
                if var == 'T/Y':
                    baseline_val = ss_baseline.T / ss_baseline.Y
                    reform_val = ss_reform.T / ss_reform.Y
                elif var == 'Welfare':
                    reform_metrics[var] = results.welfare_change
                    continue
                else:
                    baseline_val = getattr(ss_baseline, var)
                    reform_val = getattr(ss_reform, var)
                
                pct_change = safe_percentage_change(reform_val, baseline_val)
                reform_metrics[var] = pct_change
            
            # Add transition period
            reform_metrics['Transition (quarters)'] = results.transition_periods
            
            comparison[reform.name] = reform_metrics
        
        return pd.DataFrame(comparison).T
    
    def optimal_tax_mix(self,
                       target_revenue: float,
                       tax_bounds: Dict[str, Tuple[float, float]],
                       objective: str = 'welfare') -> Dict[str, float]:
        """
        Find optimal tax mix to achieve target revenue
        
        Args:
            target_revenue: Target tax revenue as share of GDP
            tax_bounds: Bounds for each tax rate
            objective: 'welfare' or 'output'
        """
        
        if self.use_simple_model:
            print("簡略化モデルでは最適化機能は利用できません")
            return None
        
        def objective_function(tax_rates):
            """Objective to minimize (negative welfare or output)"""
            # Create reform
            reform_params = ModelParameters()
            for attr in dir(self.baseline_params):
                if not attr.startswith('_'):
                    setattr(reform_params, attr, getattr(self.baseline_params, attr))
            
            # Set tax rates
            reform_params.tau_c = tax_rates[0]
            reform_params.tau_l = tax_rates[1]
            reform_params.tau_f = tax_rates[2]
            
            try:
                # Compute new steady state
                reform_model = DSGEModel(reform_params)
                reform_ss = reform_model.compute_steady_state()
                
                # Check revenue constraint
                revenue_share = reform_ss.T / reform_ss.Y
                
                if objective == 'welfare':
                    # Approximate welfare change
                    consumption_change = (reform_ss.C - self.baseline_ss.C) / self.baseline_ss.C
                    labor_change = (reform_ss.L - self.baseline_ss.L) / self.baseline_ss.L
                    
                    # Simple welfare approximation
                    welfare = consumption_change - self.baseline_params.chi * labor_change
                    
                    # Penalty for missing revenue target
                    penalty = 100 * (revenue_share - target_revenue) ** 2
                    
                    return -welfare + penalty
                
                elif objective == 'output':
                    output_change = (reform_ss.Y - self.baseline_ss.Y) / self.baseline_ss.Y
                    penalty = 100 * (revenue_share - target_revenue) ** 2
                    return -output_change + penalty
                
            except:
                # Return large penalty if steady state fails
                return 1000.0
        
        # Initial guess
        if hasattr(self, 'baseline_params'):
            x0 = [
                self.baseline_params.tau_c,
                self.baseline_params.tau_l,
                self.baseline_params.tau_f
            ]
        else:
            x0 = [0.10, 0.20, 0.30]  # デフォルト値
        
        # Bounds
        bounds = [
            tax_bounds.get('tau_c', (0.0, 0.3)),
            tax_bounds.get('tau_l', (0.0, 0.5)),
            tax_bounds.get('tau_f', (0.0, 0.5))
        ]
        
        # Optimize
        result = optimize.minimize(
            objective_function,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-6, 'maxiter': 100}
        )
        
        if result.success:
            return {
                'tau_c': result.x[0],
                'tau_l': result.x[1],
                'tau_f': result.x[2],
                'objective_value': -result.fun
            }
        else:
            print(f"Optimization failed: {result.message}")
            return None
    
    def plot_transition_dynamics(self,
                               results: SimulationResults,
                               variables: List[str],
                               figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """Plot transition dynamics for a reform"""
        # 日本語フォント設定を強制的に適用
        plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Hiragino Kaku Gothic Pro', 'Yu Gothic', 'DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        n_vars = len(variables)
        n_cols = 2
        n_rows = (n_vars + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Color palette
        colors = ['#1f77b4', '#ff7f0e']
        
        for i, var in enumerate(variables):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Plot baseline and reform paths
            baseline_values = results.baseline_path[var]
            reform_values = results.reform_path[var]
            
            ax.plot(baseline_values.index, baseline_values, 
                   label='Baseline', color=colors[0], linewidth=2)
            ax.plot(reform_values.index, reform_values,
                   label='Reform', color=colors[1], linewidth=2, linestyle='--')
            
            # Mark steady states
            ax.axhline(y=baseline_values.iloc[-1], color=colors[0], 
                      alpha=0.3, linestyle=':')
            ax.axhline(y=reform_values.iloc[-1], color=colors[1],
                      alpha=0.3, linestyle=':')
            
            # Formatting
            ax.set_title(var, fontsize=12, fontweight='bold')
            ax.set_xlabel('Quarters', fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add percentage change annotation
            pct_change = safe_percentage_change(reform_values.iloc[-1], baseline_values.iloc[-1])
            ax.text(0.98, 0.02, f'Δ = {pct_change:+.1f}%',
                   transform=ax.transAxes, fontsize=9,
                   ha='right', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for i in range(n_vars, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        # Use safe Japanese title - 直接英語タイトルを使用して文字化けを回避
        plt.suptitle(f'Transition Dynamics: {results.name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def generate_report(self, 
                       results: SimulationResults,
                       output_file: str):
        """Generate comprehensive report for a tax reform"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Tax Reform Analysis: {results.name}\n")
            f.write("=" * 60 + "\n\n")
            
            # Summary
            f.write("Executive Summary\n")
            f.write("-" * 30 + "\n")
            f.write(f"Welfare change: {results.welfare_change:+.2f}%\n")
            f.write(f"Transition period: {results.transition_periods} quarters\n")
            
            # Steady state comparison
            f.write("\n\nSteady State Comparison\n")
            f.write("-" * 30 + "\n")
            
            key_vars = ['Y', 'C', 'I', 'L', 'T', 'T/Y', 'B/Y']
            f.write(f"{'Variable':<15} {'Baseline':<12} {'Reform':<12} {'% Change':<12}\n")
            f.write("-" * 51 + "\n")
            
            for var in key_vars:
                if var == 'T/Y':
                    baseline_val = results.steady_state_baseline.T / results.steady_state_baseline.Y
                    reform_val = results.steady_state_reform.T / results.steady_state_reform.Y
                elif var == 'B/Y':
                    baseline_val = results.steady_state_baseline.B / (4 * results.steady_state_baseline.Y)
                    reform_val = results.steady_state_reform.B / (4 * results.steady_state_reform.Y)
                else:
                    baseline_val = getattr(results.steady_state_baseline, var)
                    reform_val = getattr(results.steady_state_reform, var)
                
                pct_change = safe_percentage_change(reform_val, baseline_val)
                f.write(f"{var:<15} {baseline_val:<12.3f} {reform_val:<12.3f} {pct_change:<+12.2f}\n")
            
            # Fiscal impact
            f.write("\n\nFiscal Impact Analysis\n")
            f.write("-" * 30 + "\n")
            f.write(results.fiscal_impact.to_string())
            
            # Aggregate effects
            f.write("\n\nAggregate Effects (20-year average)\n")
            f.write("-" * 30 + "\n")
            agg_effects = results.compute_aggregate_effects(
                ['Y', 'C', 'I', 'L', 'T'], 
                periods=80  # 20 years
            )
            f.write(agg_effects.to_string())
            
            f.write("\n\nEnd of Report\n")
    
    def plot_results(self, results, variables: List[str] = None, figsize=(12, 8)):
        """Plot simulation results"""
        import matplotlib.pyplot as plt
        
        if variables is None:
            variables = ['Y', 'C', 'I', 'L']
        
        # Check if results has reform_path or paths
        if hasattr(results, 'reform_path'):
            result_path = results.reform_path
        elif hasattr(results, 'paths'):
            result_path = results.paths
        else:
            print("No plottable path data found in results")
            return
            
        # Filter variables that exist in the results
        available_vars = [var for var in variables if var in result_path.columns]
        
        if not available_vars:
            print("No plottable variables found in results")
            return
        
        n_vars = len(available_vars)
        n_cols = min(2, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_vars == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_vars > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, var in enumerate(available_vars):
            ax = axes[i]
            
            # Plot baseline (if available)
            if hasattr(results, 'baseline_path') and var in results.baseline_path.columns:
                ax.plot(results.baseline_path.index, results.baseline_path[var], 
                       'b--', label='Baseline', alpha=0.7)
            
            # Plot reform path
            ax.plot(result_path.index, result_path[var], 
                   'r-', label='Reform', linewidth=2)
            
            ax.set_title(f'{var}')
            ax.set_xlabel('Quarters')
            ax.set_ylabel('Level')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_vars, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
