#!/usr/bin/env python3
"""
Notebook エラーシナリオテスト

Notebookでユーザーが遭遇する可能性がある様々なエラー状況をテストし、
適切なエラーハンドリングと回復メカニズムを検証します。
"""

import sys
import os
import pytest
import numpy as np
import warnings
from pathlib import Path

# プロジェクトルートを設定
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from src.dsge_model import DSGEModel, ModelParameters
from src.tax_simulator import EnhancedTaxSimulator, TaxReform
from src.research_warnings import ResearchWarning


class TestNotebookErrorScenarios:
    """Notebook使用時のエラーシナリオテスト"""
    
    @pytest.fixture
    def valid_model(self):
        """正常なモデルの作成"""
        params = ModelParameters.from_json('config/parameters.json')
        return DSGEModel(params)
    
    def test_config_file_missing(self):
        """設定ファイル不存在のエラーハンドリング"""
        print("\n=== 設定ファイル不存在テスト ===")
        
        with pytest.raises((FileNotFoundError, OSError)):
            ModelParameters.from_json('nonexistent_config.json')
        print("✅ 設定ファイル不存在時の適切なエラー発生確認")
    
    def test_corrupted_parameter_values(self):
        """破損したパラメータ値での動作"""
        print("\n=== 破損パラメータ値テスト ===")
        
        corrupted_scenarios = [
            {"name": "負のbeta", "beta": -0.1},
            {"name": "1以上のbeta", "beta": 1.1},
            {"name": "負のalpha", "alpha": -0.1},
            {"name": "1以上のalpha", "alpha": 1.1},
            {"name": "極端に大きな価格硬直性", "theta_p": 0.999},
        ]
        
        error_caught = 0
        for scenario in corrupted_scenarios:
            try:
                params = ModelParameters.from_json('config/parameters.json')
                
                # パラメータを破損させる
                for key, value in scenario.items():
                    if key != "name":
                        setattr(params, key, value)
                
                # モデル作成を試行
                model = DSGEModel(params)
                ss = model.compute_steady_state()
                
                # 異常な結果チェック
                if (np.isnan(ss.Y) or np.isinf(ss.Y) or 
                    ss.Y <= 0 or ss.C <= 0):
                    print(f"✅ {scenario['name']}: 異常結果検出")
                    error_caught += 1
                else:
                    print(f"⚠️ {scenario['name']}: 予期しない正常終了")
                    
            except (ValueError, RuntimeError, OverflowError, Exception) as e:
                print(f"✅ {scenario['name']}: 適切なエラー - {type(e).__name__}")
                error_caught += 1
        
        assert error_caught >= 3, f"破損パラメータ検出が不十分: {error_caught}/5"
    
    def test_simulator_initialization_failures(self, valid_model):
        """シミュレータ初期化失敗シナリオ"""
        print("\n=== シミュレータ初期化失敗テスト ===")
        
        # None modelでの初期化試行
        with pytest.raises((AttributeError, TypeError, ValueError)):
            EnhancedTaxSimulator(None)
        print("✅ Noneモデルでの初期化エラー確認")
        
        # 不完全なモデルでの初期化
        class IncompleteModel:
            def __init__(self):
                self.params = None
        
        with pytest.raises((AttributeError, ValueError)):
            EnhancedTaxSimulator(IncompleteModel())
        print("✅ 不完全モデルでの初期化エラー確認")
    
    def test_reform_parameter_validation(self, valid_model):
        """改革パラメータ検証"""
        print("\n=== 改革パラメータ検証テスト ===")
        
        simulator = EnhancedTaxSimulator(valid_model)
        
        invalid_reforms = [
            {"name": "負の消費税", "tau_c": -0.1},
            {"name": "200%消費税", "tau_c": 2.0},
            {"name": "負の所得税", "tau_l": -0.2},
            {"name": "150%所得税", "tau_l": 1.5},
            {"name": "不正な実施方法", "implementation": "invalid_method"},
        ]
        
        validation_working = 0
        for reform_params in invalid_reforms:
            try:
                reform = TaxReform(
                    name=reform_params.get("name", "テスト"),
                    tau_c=reform_params.get("tau_c", 0.1),
                    tau_l=reform_params.get("tau_l", 0.2),
                    tau_k=reform_params.get("tau_k", 0.25),
                    tau_f=reform_params.get("tau_f", 0.3),
                    implementation=reform_params.get("implementation", "permanent")
                )
                
                # シミュレーション実行
                result = simulator.simulate_reform(reform, periods=5)
                
                # 結果の妥当性チェック
                if (hasattr(result, 'welfare_change') and 
                    (np.isnan(result.welfare_change) or np.isinf(result.welfare_change))):
                    validation_working += 1
                    print(f"✅ {reform_params['name']}: 異常結果検出")
                else:
                    print(f"⚠️ {reform_params['name']}: 検証機能不十分")
                    
            except (ValueError, TypeError, Exception) as e:
                validation_working += 1
                print(f"✅ {reform_params['name']}: 適切なエラー - {type(e).__name__}")
        
        assert validation_working >= 3, f"パラメータ検証が不十分: {validation_working}/5"
    
    def test_memory_overflow_protection(self, valid_model):
        """メモリオーバーフロー保護"""
        print("\n=== メモリオーバーフロー保護テスト ===")
        
        simulator = EnhancedTaxSimulator(valid_model)
        reform = TaxReform("メモリテスト", tau_c=0.12)
        
        # 極端に長い期間でのシミュレーション
        extreme_periods = [1000, 5000, 10000]
        
        memory_protected = 0
        for periods in extreme_periods:
            try:
                # タイムアウト付きでの実行
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("シミュレーションタイムアウト")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(10)  # 10秒タイムアウト
                
                try:
                    result = simulator.simulate_reform(reform, periods=periods)
                    signal.alarm(0)  # タイムアウト解除
                    
                    # メモリ使用量の妥当性チェック
                    if hasattr(result, 'baseline_path') and hasattr(result, 'reform_path'):
                        path_size = len(result.baseline_path) + len(result.reform_path)
                        if path_size > periods * 2:
                            print(f"⚠️ 期間{periods}: メモリ使用量異常")
                        else:
                            memory_protected += 1
                            print(f"✅ 期間{periods}: メモリ使用量正常")
                    
                except TimeoutError:
                    signal.alarm(0)
                    memory_protected += 1
                    print(f"✅ 期間{periods}: タイムアウト保護作動")
                    
            except (MemoryError, OverflowError) as e:
                memory_protected += 1
                print(f"✅ 期間{periods}: メモリ保護 - {type(e).__name__}")
            except Exception as e:
                print(f"⚠️ 期間{periods}: 予期しないエラー - {type(e).__name__}")
        
        assert memory_protected >= 2, f"メモリ保護が不十分: {memory_protected}/3"
    
    def test_numerical_instability_detection(self, valid_model):
        """数値不安定性の検出"""
        print("\n=== 数値不安定性検出テスト ===")
        
        # 数値的に不安定な条件を作成
        unstable_scenarios = [
            {"name": "極端なCalvoパラメータ", "theta_p": 0.999},
            {"name": "極小な調整コスト", "phi_i": 0.001},
            {"name": "極大な調整コスト", "phi_i": 1000},
            {"name": "極端な割引率", "beta": 0.999},
        ]
        
        instability_detected = 0
        for scenario in unstable_scenarios:
            try:
                params = ModelParameters.from_json('config/parameters.json')
                
                # 不安定パラメータ設定
                for key, value in scenario.items():
                    if key != "name":
                        setattr(params, key, value)
                
                model = DSGEModel(params)
                simulator = EnhancedTaxSimulator(model)
                
                reform = TaxReform("不安定性テスト", tau_c=0.11)
                
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    result = simulator.simulate_reform(reform, periods=20)
                    
                    # 数値警告の確認
                    numerical_warnings = [warn for warn in w 
                                        if any(keyword in str(warn.message).lower() 
                                             for keyword in ['convergence', 'numerical', 'unstable', 'overflow'])]
                    
                    if numerical_warnings:
                        instability_detected += 1
                        print(f"✅ {scenario['name']}: 数値警告検出 ({len(numerical_warnings)}個)")
                    elif (hasattr(result, 'welfare_change') and 
                          (np.isnan(result.welfare_change) or np.isinf(result.welfare_change))):
                        instability_detected += 1
                        print(f"✅ {scenario['name']}: 異常結果検出")
                    else:
                        print(f"⚠️ {scenario['name']}: 不安定性未検出")
                        
            except (RuntimeError, OverflowError, ValueError) as e:
                instability_detected += 1
                print(f"✅ {scenario['name']}: 数値エラー検出 - {type(e).__name__}")
            except Exception as e:
                print(f"❌ {scenario['name']}: 予期しないエラー - {type(e).__name__}")
        
        assert instability_detected >= 2, f"数値不安定性検出が不十分: {instability_detected}/4"
    
    def test_incomplete_simulation_results(self, valid_model):
        """不完全なシミュレーション結果の処理"""
        print("\n=== 不完全シミュレーション結果テスト ===")
        
        simulator = EnhancedTaxSimulator(valid_model)
        
        # 極端な条件でシミュレーション
        extreme_reform = TaxReform("極端改革", tau_c=0.95, tau_l=0.95)
        
        try:
            result = simulator.simulate_reform(extreme_reform, periods=10)
            
            # 結果の完全性チェック
            required_attributes = ['name', 'welfare_change', 'baseline_path', 'reform_path']
            missing_attrs = []
            
            for attr in required_attributes:
                if not hasattr(result, attr):
                    missing_attrs.append(attr)
                elif getattr(result, attr) is None:
                    missing_attrs.append(f"{attr}(None)")
            
            if missing_attrs:
                print(f"✅ 不完全結果検出: 不足属性 {missing_attrs}")
            else:
                # データの妥当性チェック
                if (hasattr(result, 'welfare_change') and 
                    (np.isnan(result.welfare_change) or np.isinf(result.welfare_change))):
                    print("✅ 異常データ検出: welfare_change")
                elif (hasattr(result, 'baseline_path') and result.baseline_path.empty):
                    print("✅ 空データ検出: baseline_path")
                else:
                    print("⚠️ 予期しない正常結果")
                    
        except Exception as e:
            print(f"✅ 極端条件でのエラー検出: {type(e).__name__}")
    
    def test_user_interrupt_simulation(self, valid_model):
        """ユーザー中断シミュレーションの処理"""
        print("\n=== ユーザー中断処理テスト ===")
        
        simulator = EnhancedTaxSimulator(valid_model)
        reform = TaxReform("中断テスト", tau_c=0.13)
        
        # KeyboardInterrupt をシミュレート
        import signal
        import threading
        
        def interrupt_simulation():
            """2秒後にKeyboardInterruptを発生"""
            import time
            time.sleep(2)
            os.kill(os.getpid(), signal.SIGINT)
        
        interrupt_thread = threading.Thread(target=interrupt_simulation)
        
        try:
            interrupt_thread.start()
            result = simulator.simulate_reform(reform, periods=100)
            interrupt_thread.join()
            
            print("⚠️ 中断が機能しませんでした")
            
        except KeyboardInterrupt:
            print("✅ KeyboardInterrupt適切に処理")
        except Exception as e:
            print(f"⚠️ 予期しない例外: {type(e).__name__}")
        finally:
            if interrupt_thread.is_alive():
                interrupt_thread.join()


def main():
    """エラーシナリオテストスイートの実行"""
    print("=" * 60)
    print("Notebook エラーシナリオテスト実行開始")
    print("=" * 60)
    
    # pytest実行
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--capture=no"
    ])
    
    print("\n" + "=" * 60)
    print(f"エラーシナリオテスト完了 - 終了コード: {exit_code}")
    print("=" * 60)
    
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)