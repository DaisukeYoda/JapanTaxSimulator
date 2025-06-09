"""
Notebook共通インフラストラクチャ
Issue #34: 教育・研究・政策分析用の統合的学習環境構築

このモジュールは全notebookで共通使用される機能を提供します：
- 環境初期化
- 研究整合性チェック
- エラーハンドリング
- 可視化ユーティリティ
"""

import sys
import os
import warnings
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 日本語フォント警告を抑制
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')
warnings.filterwarnings('ignore', message='.*Glyph.*missing from font.*')

# 日本語フォント用の便利なインポート
__all__ = [
    'setup_notebook_environment', 
    'load_baseline_model', 
    'create_research_simulator',
    'get_japanese_font',
    'apply_japanese_font_to_axes',
    'plot_scenario_comparison',
    'safe_simulation_wrapper',
    'validate_research_compliance'
]

# プロジェクト固有モジュール（遅延インポート用）
# from src.dsge_model import DSGEModel, ModelParameters, load_model
# from src.tax_simulator import ResearchTaxSimulator, EnhancedTaxSimulator, TaxReform  
# from src.plot_utils import setup_plotting_style


def setup_notebook_environment(notebook_name: str = "notebook") -> str:
    """
    Notebook実行環境の統一設定
    
    Args:
        notebook_name: ノートブック名（ログ用）
        
    Returns:
        project_root: プロジェクトルートパス
    """
    # 確実なプロジェクトルート設定
    current_dir = os.getcwd()
    
    # notebooksディレクトリから実行される場合
    if current_dir.endswith('notebooks'):
        project_root = os.path.dirname(current_dir)
        os.chdir(project_root)
    else:
        # JapanTaxSimulatorディレクトリを探す
        while not os.path.exists('config/parameters.json') and os.getcwd() != '/':
            if 'JapanTaxSimulator' in os.getcwd():
                break
            os.chdir('..')
        project_root = os.getcwd()
    
    # 設定ファイル確認
    if not os.path.exists('config/parameters.json'):
        raise FileNotFoundError(f"設定ファイルが見つかりません: {project_root}/config/parameters.json")
    
    # sys.pathに追加
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # RESEARCH_MODE環境変数設定（警告回避）
    if 'RESEARCH_MODE' not in os.environ:
        os.environ['RESEARCH_MODE'] = 'development'
    
    # 日本語フォント強制設定
    _force_japanese_font_setup()
    
    print(f"📁 プロジェクトルート: {project_root}")
    print(f"📋 {notebook_name} 環境設定完了")
    
    return project_root


def _force_japanese_font_setup():
    """notebook用日本語フォント強制設定"""
    import matplotlib.font_manager as fm
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Get Hiragino font path directly
        hiragino_path = None
        for font in fm.fontManager.ttflist:
            if font.name == 'Hiragino Sans' and 'ヒラギノ角ゴシック' in font.fname:
                hiragino_path = font.fname
                break
        
        if hiragino_path:
            # Force matplotlib to use the exact font
            hiragino_prop = fm.FontProperties(fname=hiragino_path)
            plt.rcParams.update({
                'font.family': 'sans-serif',
                'font.sans-serif': ['Hiragino Sans'] + plt.rcParams['font.sans-serif'],
                'axes.unicode_minus': False,
                'font.size': 12
            })
            
            # Force font cache rebuild
            try:
                fm.fontManager.addfont(hiragino_path)
                fm._rebuild()
            except:
                pass
                
            print(f"🎌 日本語フォント強制設定: Hiragino Sans ({hiragino_path})")
        else:
            # Fallback approach
            plt.rcParams.update({
                'font.family': 'sans-serif',
                'font.sans-serif': ['Hiragino Sans', 'Hiragino Kaku Gothic Pro'] + plt.rcParams['font.sans-serif'],
                'axes.unicode_minus': False,
                'font.size': 12
            })
            print("🎌 日本語フォント設定: Hiragino Sans (fallback)")


class NotebookEnvironment:
    """Notebook実行環境の管理クラス（非推奨：setup_notebook_environment使用推奨）"""
    
    def __init__(self, notebook_name: str, research_mode: bool = True):
        """
        Args:
            notebook_name: ノートブック名（ログ用）
            research_mode: 研究モード
        """
        self.notebook_name = notebook_name
        self.research_mode = research_mode
        self.project_root = setup_notebook_environment(notebook_name)
    
    def setup_environment(self) -> Dict[str, Any]:
        """notebook環境の完全初期化"""
        print(f"🚀 {self.notebook_name} 環境初期化中...")
        
        # 日本語フォント設定
        try:
            setup_plotting_style()
            print("✅ 日本語フォント設定完了")
        except Exception as e:
            print(f"⚠️ フォント設定警告: {e}")
        
        # プロジェクト構成確認
        config_path = 'config/parameters.json'
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
        
        print(f"✅ プロジェクトルート: {self.project_root}")
        print(f"✅ 設定ファイル確認済み")
        
        return {
            'project_root': self.project_root,
            'config_path': config_path,
            'research_mode': self.research_mode
        }


def load_baseline_model(config_path: str = 'config/parameters.json'):
    """
    ベースラインモデルの読み込み（定常状態計算込み）
    """
    # 遅延インポート
    from src.dsge_model import load_model
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
    
    try:
        model = load_model(config_path)
        print("✅ ベースラインモデル読み込み成功")
        
        # 定常状態が計算されていない場合は計算
        if not hasattr(model, 'steady_state') or model.steady_state is None:
            print("🔄 定常状態を計算中...")
            steady_state = model.compute_steady_state()
            print("✅ 定常状態計算完了")
        else:
            print("✅ 定常状態は既に計算済み")
            
        return model
    except Exception as e:
        raise RuntimeError(f"モデル初期化失敗: {e}")


def create_research_simulator(
    model, 
    force_research_mode: bool = True,
    use_simple_linearization: bool = False
):
    """
    研究グレードシミュレータの作成
    🚨 CRITICAL: DummySteadyState絶対禁止
    """
    # 遅延インポート
    from src.tax_simulator import ResearchTaxSimulator, EnhancedTaxSimulator
    
    status_info = {
        'simulator_type': '',
        'linearization_method': '',
        'research_compliance': '',
        'warnings': ''
    }
    
    # 研究グレードシミュレータを最優先で試行
    try:
        simulator = ResearchTaxSimulator(
            baseline_model=model,
            use_simple_linearization=use_simple_linearization
        )
        status_info['simulator_type'] = 'ResearchTaxSimulator'
        status_info['linearization_method'] = 'simplified' if use_simple_linearization else 'full_klein'
        status_info['research_compliance'] = '✅ COMPLIANT'
        print("🎓 研究グレードシミュレータ初期化成功")
        return simulator, status_info
        
    except Exception as e:
        status_info['warnings'] += f"ResearchTaxSimulator失敗: {e}; "
        print(f"⚠️ 研究グレードシミュレータ初期化失敗: {e}")
    
    # 拡張シミュレータをフォールバック（DummySteadyState完全回避）
    try:
        simulator = EnhancedTaxSimulator(
            model, 
            use_simple_model=False,  # 🚨 DummySteadyState完全回避
            research_mode=True       # 🚨 常に研究モード強制
        )
        status_info['simulator_type'] = 'EnhancedTaxSimulator'
        status_info['linearization_method'] = 'enhanced'
        status_info['research_compliance'] = '✅ COMPLIANT'
        print("📚 拡張シミュレータ初期化成功（DummySteadyState回避）")
        return simulator, status_info
        
    except Exception as e:
        status_info['warnings'] += f"EnhancedTaxSimulator失敗: {e}; "
        print(f"❌ 拡張シミュレータ初期化失敗: {e}")
    
    # 🚨 DummySteadyStateを使用する可能性のあるフォールバックは完全削除
    # 代わりに明確なエラーで失敗
    raise RuntimeError(
        f"🚨 CRITICAL: 研究整合性を保つシミュレータの初期化に失敗しました。\n"
        f"DummySteadyStateは絶対に使用しません。\n"
        f"エラー詳細: {status_info['warnings']}\n"
        f"解決策: モデルパラメータまたは計算設定を確認してください。"
    )


def validate_research_compliance(simulator) -> Dict[str, Any]:
    """
    研究整合性の検証
    
    Args:
        simulator: 検証対象シミュレータ
        
    Returns:
        検証結果辞書
    """
    validation_result = {
        'is_research_compliant': False,
        'simulator_type': type(simulator).__name__,
        'dummy_state_risk': 'UNKNOWN',
        'linearization_method': 'UNKNOWN',
        'warnings': []
    }
    
    try:
        # 遅延インポートで型チェック
        from src.tax_simulator import ResearchTaxSimulator
        
        # ResearchTaxSimulatorチェック
        if isinstance(simulator, ResearchTaxSimulator):
            validation_result['is_research_compliant'] = True
            validation_result['dummy_state_risk'] = 'NONE'
            validation_result['linearization_method'] = 'KLEIN_METHOD'
        
        # EnhancedTaxSimulatorチェック
        elif hasattr(simulator, 'research_mode') and simulator.research_mode:
            validation_result['is_research_compliant'] = True
            validation_result['dummy_state_risk'] = 'LOW'
        elif hasattr(simulator, 'use_simple_model') and not simulator.use_simple_model:
            validation_result['is_research_compliant'] = True
            validation_result['dummy_state_risk'] = 'LOW'
        else:
            validation_result['warnings'].append('DummySteadyState使用の可能性')
            validation_result['dummy_state_risk'] = 'HIGH'
        
        # 線形化手法チェック
        if hasattr(simulator, 'use_simple_linearization'):
            if simulator.use_simple_linearization:
                validation_result['linearization_method'] = 'SIMPLIFIED'
                validation_result['warnings'].append('簡略化線形化使用中')
            else:
                validation_result['linearization_method'] = 'FULL_KLEIN'
        
    except Exception as e:
        validation_result['warnings'].append(f'検証エラー: {e}')
    
    return validation_result


def safe_simulation_wrapper(
    simulator, 
    reform, 
    periods: int = 40,
    research_mode: bool = True
):
    """
    安全なシミュレーション実行ラッパー
    
    Args:
        simulator: シミュレータ
        reform: 税制改革
        periods: シミュレーション期間
        research_mode: 研究モード
        
    Returns:
        (results, execution_info): 結果と実行情報
    """
    execution_info = {
        'status': 'PENDING',
        'error': '',
        'dummy_state_detected': 'NO',
        'welfare_available': 'NO'
    }
    
    try:
        # 研究モードでの事前チェック
        if research_mode:
            validation = validate_research_compliance(simulator)
            if not validation['is_research_compliant']:
                execution_info['error'] = f"研究整合性違反: {validation['warnings']}"
                execution_info['status'] = 'RESEARCH_VIOLATION'
                return None, execution_info
        
        # シミュレーション実行
        results = simulator.simulate_reform(reform=reform, periods=periods)
        execution_info['status'] = 'SUCCESS'
        
        # 結果の研究整合性チェック
        if hasattr(results, 'steady_state_baseline') and hasattr(results, 'steady_state_reform'):
            baseline_type = type(results.steady_state_baseline).__name__
            reform_type = type(results.steady_state_reform).__name__
            
            if 'Dummy' in baseline_type or 'Dummy' in reform_type:
                execution_info['dummy_state_detected'] = 'YES'
                execution_info['error'] = f"DummySteadyState検出: {baseline_type}, {reform_type}"
                if research_mode:
                    execution_info['status'] = 'RESEARCH_VIOLATION'
                    return None, execution_info
        
        # 福利厚生分析チェック
        if hasattr(results, 'welfare_change'):
            execution_info['welfare_available'] = 'YES'
        
        return results, execution_info
        
    except Exception as e:
        execution_info['status'] = 'ERROR'
        execution_info['error'] = str(e)
        return None, execution_info


def create_comparison_dataframe(
    scenarios, 
    baseline_model
):
    """
    シナリオ比較用データフレームの作成
    """
    # 遅延インポート
    from src.dsge_model import DSGEModel, ModelParameters
    
    results_summary = pd.DataFrame()
    
    for scenario_name, tax_rates in scenarios.items():
        try:
            # パラメータを複製
            scenario_params = ModelParameters()
            for attr in dir(baseline_model.params):
                if not attr.startswith('_'):
                    setattr(scenario_params, attr, getattr(baseline_model.params, attr))
            
            # 税率を更新
            for tax, rate in tax_rates.items():
                setattr(scenario_params, tax, rate)
            
            # モデルを作成し定常状態を計算
            scenario_model = DSGEModel(scenario_params)
            scenario_ss = scenario_model.compute_steady_state()
            
            # 結果を記録
            results = {
                'GDP': scenario_ss.Y,
                '消費': scenario_ss.C,
                '投資': scenario_ss.I,
                '労働': scenario_ss.L,
            }
            
            # 税収データの安全な取得
            total_revenue = getattr(scenario_ss, 'T_total_revenue', getattr(scenario_ss, 'T', None))
            if total_revenue is not None:
                results['総税収'] = total_revenue
                results['税収/GDP'] = total_revenue / scenario_ss.Y
            
            results_summary[scenario_name] = pd.Series(results)
            print(f"✅ {scenario_name}: 計算成功")
            
        except Exception as e:
            print(f"❌ {scenario_name}: 計算失敗 - {e}")
    
    return results_summary


def plot_scenario_comparison(results_df: pd.DataFrame, figsize: Tuple[int, int] = (12, 10)):
    """
    シナリオ比較の可視化
    
    Args:
        results_df: 比較結果DataFrame
        figsize: 図のサイズ
    """
    if results_df.empty:
        print("⚠️ プロット用データが空です")
        return
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # 日本語フォント強制再設定
        _force_japanese_font_setup()
        
        # Get Japanese font properties
        jp_font = _get_japanese_font_prop()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # GDP比較
        if 'GDP' in results_df.index:
            ax = axes[0, 0]
            results_df.loc['GDP'].plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title('GDP水準の比較', fontproperties=jp_font)
            ax.set_ylabel('GDP', fontproperties=jp_font)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        # 税収/GDP比率
        if '税収/GDP' in results_df.index:
            ax = axes[0, 1]
            (results_df.loc['税収/GDP'] * 100).plot(kind='bar', ax=ax, color='coral')
            ax.set_title('税収/GDP比率の比較', fontproperties=jp_font)
            ax.set_ylabel('税収/GDP (%)', fontproperties=jp_font)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        # 消費と投資
        if all(var in results_df.index for var in ['消費', '投資']):
            ax = axes[1, 0]
            results_df.loc[['消費', '投資']].T.plot(kind='bar', ax=ax)
            ax.set_title('消費と投資の比較', fontproperties=jp_font)
            ax.set_ylabel('水準', fontproperties=jp_font)
            ax.tick_params(axis='x', rotation=45)
            ax.legend(['消費', '投資'], prop=jp_font)
            ax.grid(True, alpha=0.3)
        
        # 労働時間
        if '労働' in results_df.index:
            ax = axes[1, 1]
            results_df.loc['労働'].plot(kind='bar', ax=ax, color='lightgreen')
            ax.set_title('労働時間の比較', fontproperties=jp_font)
            ax.set_ylabel('労働時間', fontproperties=jp_font)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def _get_japanese_font_prop():
    """日本語フォントプロパティを取得"""
    import matplotlib.font_manager as fm
    
    # Try to find Hiragino Sans font path
    for font in fm.fontManager.ttflist:
        if font.name == 'Hiragino Sans' and 'ヒラギノ角ゴシック' in font.fname:
            return fm.FontProperties(fname=font.fname)
    
    # Fallback: try by name
    try:
        return fm.FontProperties(family='Hiragino Sans')
    except:
        return fm.FontProperties(family='sans-serif')


def get_japanese_font():
    """
    Notebook用日本語フォントプロパティ取得
    03 notebookなどで直接使用可能
    """
    return _get_japanese_font_prop()


def apply_japanese_font_to_axes(ax, jp_font=None):
    """
    Axesオブジェクトに日本語フォントを適用
    
    Args:
        ax: matplotlib Axes object
        jp_font: FontProperties (None の場合は自動取得)
    """
    if jp_font is None:
        jp_font = _get_japanese_font_prop()
    
    # タイトルとラベルにフォントを適用
    if ax.get_title():
        ax.set_title(ax.get_title(), fontproperties=jp_font)
    if ax.get_xlabel():
        ax.set_xlabel(ax.get_xlabel(), fontproperties=jp_font) 
    if ax.get_ylabel():
        ax.set_ylabel(ax.get_ylabel(), fontproperties=jp_font)
    
    # 凡例にもフォントを適用
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontproperties(jp_font)


def print_research_disclaimer():
    """研究使用に関する免責事項の表示"""
    print("\n" + "="*60)
    print("🎓 研究使用に関する重要な注意事項")
    print("="*60)
    print("• このモデルは学術研究・政策分析用です")
    print("• 結果は特定の仮定に基づくシミュレーションです")
    print("• 実際の政策決定には追加の実証分析が必要です")
    print("• パラメータの不確実性を考慮した感度分析を推奨します")
    print("• 引用時は使用した線形化手法を明記してください")
    print("="*60 + "\n")


# 教育用のサンプルシナリオ
SAMPLE_TAX_SCENARIOS = {
    'ベースライン': {'tau_c': 0.10, 'tau_l': 0.20, 'tau_f': 0.30},
    '消費税15%': {'tau_c': 0.15, 'tau_l': 0.20, 'tau_f': 0.30},
    '所得税15%': {'tau_c': 0.10, 'tau_l': 0.15, 'tau_f': 0.30},
    '法人税25%': {'tau_c': 0.10, 'tau_l': 0.20, 'tau_f': 0.25},
    '複合改革': {'tau_c': 0.15, 'tau_l': 0.15, 'tau_f': 0.30}
}

# 主要変数の日本語マッピング
VARIABLE_NAMES_JP = {
    'Y': 'GDP',
    'C': '消費',
    'I': '投資',
    'L': '労働時間',
    'w': '実質賃金',
    'r': '実質利子率',
    'pi': 'インフレ率',
    'G': '政府支出',
    'T': '税収',
    'B': '政府債務'
}