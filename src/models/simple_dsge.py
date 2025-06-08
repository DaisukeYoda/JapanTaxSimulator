#!/usr/bin/env python3
"""
税制シミュレーション用の簡略化DSGEモデルを作成

⚠️ RESEARCH WARNING: This is a SIMPLIFIED model with different economic assumptions
than the full DSGE model. Results are NOT COMPARABLE and should not be used for research
without explicit validation against empirical data.
"""

import sys
import numpy as np
import json
import warnings
from dataclasses import dataclass
from typing import Dict, Optional
from scipy import optimize
import os

# Add project root to path (src/models -> project root)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import research warnings
try:
    from src.research_warnings import (
        research_critical, 
        research_deprecated, 
        research_requires_validation,
        check_research_mode,
        ResearchWarning
    )
    
    # Check research mode
    check_research_mode()
    
    # Warn about simplified model
    warnings.warn(
        "🚨 SIMPLIFIED DSGE MODEL: This model uses only 8 variables and simplified equations. "
        "Economic assumptions differ significantly from full DSGE models. "
        "Results should NOT be used for research without empirical validation.",
        ResearchWarning
    )
    
except ImportError:
    # Define minimal warning if research_warnings not available
    class ResearchWarning(UserWarning):
        pass
    
    def research_critical(message):
        def decorator(func):
            def wrapper(*args, **kwargs):
                warnings.warn(f"RESEARCH CRITICAL: {message}", ResearchWarning)
                return func(*args, **kwargs)
            return wrapper
        return decorator

@dataclass
class SimpleDSGEParameters:
    """簡略化DSGEモデルのパラメータ"""
    # 家計
    beta: float = 0.99      # 割引因子
    sigma_c: float = 1.5    # 消費の異時点間代替弾力性
    sigma_l: float = 2.0    # 労働供給弾力性
    chi: float = 1.0        # 労働不効用
    habit: float = 0.3      # 習慣形成
    
    # 企業
    alpha: float = 0.33     # 資本分配率
    delta: float = 0.025    # 減価償却率
    
    # 政府
    gy_ratio: float = 0.20  # 政府支出/GDP比
    
    # 税率
    tau_c: float = 0.10     # 消費税
    tau_l: float = 0.20     # 所得税
    tau_k: float = 0.25     # 資本税
    
    @classmethod
    def from_config(cls, override_tau_c=None, override_tau_l=None, override_tau_k=None):
        """設定ファイルから読み込み（税率オーバーライド可能）"""
        try:
            config_path = os.path.join(project_root, 'config', 'parameters.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            household = config['model_parameters']['household']
            firm = config['model_parameters']['firm']
            gov = config['model_parameters']['government']
            tax = config['tax_parameters']['baseline']
            
            # 税率のオーバーライド
            tau_c = override_tau_c if override_tau_c is not None else tax['tau_c']
            tau_l = override_tau_l if override_tau_l is not None else tax['tau_l']
            tau_k = override_tau_k if override_tau_k is not None else tax['tau_k']
            
            return cls(
                beta=household['beta'],
                sigma_c=household['sigma_c'],
                sigma_l=household['sigma_l'],
                chi=household['chi'],
                habit=household['habit'],
                alpha=firm['alpha'],
                delta=firm['delta'],
                gy_ratio=gov['gy_ratio'],
                tau_c=tau_c,
                tau_l=tau_l,
                tau_k=tau_k
            )
        except Exception as e:
            print(f"設定読み込みエラー、デフォルト値使用: {e}")
            return cls(
                tau_c=override_tau_c if override_tau_c is not None else 0.10,
                tau_l=override_tau_l if override_tau_l is not None else 0.20,
                tau_k=override_tau_k if override_tau_k is not None else 0.25
            )

@dataclass
class SimpleSteadyState:
    """簡略化された定常状態"""
    Y: float = 0.0          # GDP
    C: float = 0.0          # 消費
    I: float = 0.0          # 投資
    K: float = 0.0          # 資本
    L: float = 0.0          # 労働
    w: float = 0.0          # 賃金
    r: float = 0.0          # 実質利子率
    Lambda: float = 0.0     # 限界効用
    G: float = 0.0          # 政府支出
    T: float = 0.0          # 総税収
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'Y': self.Y, 'C': self.C, 'I': self.I, 'K': self.K, 'L': self.L,
            'w': self.w, 'r': self.r, 'Lambda': self.Lambda, 'G': self.G, 'T': self.T
        }

class SimpleDSGEModel:
    """税制シミュレーション用簡略化DSGEモデル"""
    
    def __init__(self, params: SimpleDSGEParameters):
        self.params = params
        self.steady_state: Optional[SimpleSteadyState] = None
        
        # 求解対象の変数
        self.variables = ['Y', 'C', 'I', 'K', 'L', 'w', 'r', 'Lambda']
    
    def steady_state_equations(self, x: np.ndarray) -> np.ndarray:
        """定常状態方程式（コア8変数のみ）"""
        Y, C, I, K, L, w, r, Lambda = x
        params = self.params
        
        # 政府支出（外生的に決定）
        G = params.gy_ratio * Y
        
        # 税収
        T = params.tau_c * C + params.tau_l * w * L + params.tau_k * r * K
        
        # 8つの方程式
        eq1 = (1 - params.beta * params.habit) / (C * (1 - params.habit)) - Lambda * (1 + params.tau_c)
        eq2 = params.chi * L**(1/params.sigma_l) - Lambda * (1 - params.tau_l) * w / (1 + params.tau_c)
        eq3 = 1 - params.beta * (1 + (1 - params.tau_k) * r - params.delta)
        eq4 = Y - K**params.alpha * L**(1 - params.alpha)
        eq5 = w - (1 - params.alpha) * Y / L
        eq6 = r - params.alpha * Y / K
        eq7 = I - params.delta * K
        eq8 = Y - C - I - G
        
        return np.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8])
    
    def compute_steady_state(self) -> SimpleSteadyState:
        """定常状態の計算"""
        
        # 初期推定値（経済的に妥当）
        Y_guess = 1.0
        C_guess = 0.55 * Y_guess
        I_guess = 0.20 * Y_guess  
        K_guess = 4.0 * Y_guess
        L_guess = 0.33
        w_guess = 1.0
        r_guess = 0.08  # 年率約8%
        Lambda_guess = 1.0
        
        x0 = np.array([Y_guess, C_guess, I_guess, K_guess, L_guess, w_guess, r_guess, Lambda_guess])
        
        print("=== 簡略化DSGE定常状態計算 ===")
        print(f"初期推定: Y={Y_guess:.3f}, C={C_guess:.3f}, I={I_guess:.3f}, L={L_guess:.3f}")
        
        try:
            # 数値求解
            result = optimize.root(self.steady_state_equations, x0, method='hybr', 
                                 options={'xtol': 1e-8, 'maxfev': 2000})
            
            if result.success:
                Y, C, I, K, L, w, r, Lambda = result.x
                G = self.params.gy_ratio * Y
                T = self.params.tau_c * C + self.params.tau_l * w * L + self.params.tau_k * r * K
                
                # 結果の妥当性チェック
                if all(val > 0 for val in [Y, C, I, K, L, w, Lambda]) and abs(Y - C - I - G) < 1e-6:
                    ss = SimpleSteadyState(Y=Y, C=C, I=I, K=K, L=L, w=w, r=r, Lambda=Lambda, G=G, T=T)
                    self.steady_state = ss
                    
                    print("✅ 定常状態計算成功")
                    print(f"  Y={Y:.3f}, C/Y={C/Y:.1%}, I/Y={I/Y:.1%}, G/Y={G/Y:.1%}")
                    print(f"  L={L:.3f}, w={w:.3f}, r={r:.3f}")
                    print(f"  総需要バランス誤差: {abs(Y - C - I - G):.2e}")
                    
                    return ss
                else:
                    print(f"❌ 解が経済的に不適切")
                    print(f"  負値変数: {[(var, val) for var, val in zip(self.variables, result.x) if val <= 0]}")
                    print(f"  総需要バランス誤差: {abs(Y - C - I - G):.2e}")
                    
            else:
                print(f"❌ 数値求解失敗: {result.message}")
                
        except Exception as e:
            print(f"❌ 計算中エラー: {e}")
            
        return None
    
    @research_critical(
        "Simplified tax simulation using only 8-equation model. "
        "Lacks dynamic adjustment, expectations, and many macroeconomic channels. "
        "Results are approximations only and should not be used for policy analysis."
    )
    def simulate_tax_change(self, new_tau_c: Optional[float] = None, 
                           new_tau_l: Optional[float] = None,
                           new_tau_k: Optional[float] = None) -> Dict[str, float]:
        """
        税制変更の影響をシミュレーション
        
        ⚠️ RESEARCH WARNING: Highly simplified model - results are approximations only
        """
        
        if self.steady_state is None:
            print("❌ ベースライン定常状態が未計算")
            return {}
        
        # 元のパラメータを保存
        original_params = SimpleDSGEParameters(
            beta=self.params.beta, sigma_c=self.params.sigma_c, sigma_l=self.params.sigma_l,
            chi=self.params.chi, habit=self.params.habit, alpha=self.params.alpha,
            delta=self.params.delta, gy_ratio=self.params.gy_ratio,
            tau_c=self.params.tau_c, tau_l=self.params.tau_l, tau_k=self.params.tau_k
        )
        
        # ベースライン定常状態を保存
        baseline = self.steady_state.to_dict()
        
        # 税率を変更
        if new_tau_c is not None:
            self.params.tau_c = new_tau_c
        if new_tau_l is not None:
            self.params.tau_l = new_tau_l
        if new_tau_k is not None:
            self.params.tau_k = new_tau_k
        
        print(f"\n=== 税制変更シミュレーション ===")
        print(f"税率変更: τc={original_params.tau_c:.1%}→{self.params.tau_c:.1%}, " +
              f"τl={original_params.tau_l:.1%}→{self.params.tau_l:.1%}, " +
              f"τk={original_params.tau_k:.1%}→{self.params.tau_k:.1%}")
        
        # 新しい定常状態を計算
        new_ss = self.compute_steady_state()
        
        # パラメータを復元
        self.params = original_params
        
        if new_ss is None:
            print("❌ 新しい定常状態の計算失敗")
            return {}
        
        # 変化率を計算
        reform = new_ss.to_dict()
        
        changes = {}
        for var in ['Y', 'C', 'I', 'L', 'T']:
            if baseline[var] != 0:
                pct_change = (reform[var] - baseline[var]) / baseline[var] * 100
                changes[f'{var}_change_pct'] = pct_change
            else:
                changes[f'{var}_change_pct'] = 0.0
        
        # 実際の新値と基準値も記録
        changes['baseline_values'] = baseline
        changes['reform_values'] = reform
        
        print(f"主要変数への影響:")
        for var in ['Y', 'C', 'I', 'L']:
            if f'{var}_change_pct' in changes:
                print(f"  {var}: {changes[f'{var}_change_pct']:+.3f}%")
            else:
                print(f"  {var}: データなし")
        
        # 実際の値の変化も表示
        print(f"実際の値の変化:")
        for var in ['Y', 'C', 'I', 'L']:
            if var in baseline and var in reform:
                print(f"  {var}: {baseline[var]:.3f} → {reform[var]:.3f} ({(reform[var] - baseline[var]) / baseline[var] * 100:+.3f}%)")
        
        return changes

def test_simple_model():
    """簡略化モデルのテスト"""
    print("=== 簡略化DSGEモデルテスト ===")
    
    # パラメータ読み込み
    params = SimpleDSGEParameters.from_config()
    
    # モデル作成
    model = SimpleDSGEModel(params)
    
    # ベースライン定常状態
    baseline_ss = model.compute_steady_state()
    
    if baseline_ss is None:
        print("❌ ベースライン計算失敗")
        return False
    
    # 税制変更テスト
    print(f"\n=== 消費税5%引き上げテスト ===")
    changes = model.simulate_tax_change(new_tau_c=0.15)  # 10% → 15%
    
    if changes:
        print("✅ 税制シミュレーション成功")
        return True
    else:
        print("❌ 税制シミュレーション失敗")
        return False

if __name__ == "__main__":
    success = test_simple_model()
    if success:
        print("\n✅ 簡略化DSGEモデルは正常に動作します")
        print("このモデルを本システムに統合することを推奨します")
    else:
        print("\n❌ さらなる調整が必要です")