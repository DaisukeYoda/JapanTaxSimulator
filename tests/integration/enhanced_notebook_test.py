#!/usr/bin/env python3
"""
拡張Notebookテストスイート - notebook安定性向上

このテストは以下を検証:
1. 研究整合性 - DummySteadyState使用の検出
2. エラーハンドリング - 様々な失敗シナリオでの安定性
3. 境界値テスト - 極端なパラメータでの動作
4. API互換性 - 異なる設定での一貫性
5. メモリ効率性 - 長時間実行での安定性
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd
import warnings
from pathlib import Path

# プロジェクトルートを設定
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from src.dsge_model import DSGEModel, ModelParameters
from src.tax_simulator import EnhancedTaxSimulator, TaxReform
from src.research_warnings import ResearchWarning


class TestNotebookRobustness:
    """Notebook安定性のための包括的テストスイート"""
    
    @pytest.fixture(scope="class")
    def base_model(self):
        """基本モデルの設定"""
        params = ModelParameters.from_json('config/parameters.json')
        return DSGEModel(params)
    
    @pytest.fixture(scope="class") 
    def base_simulator(self, base_model):
        """基本シミュレータの設定"""
        return EnhancedTaxSimulator(base_model, use_simple_model=False)
    
    def test_research_integrity_dummy_detection(self, base_model):
        """研究整合性: DummySteadyState使用の検出"""
        print("\n=== 研究整合性テスト: DummySteadyState検出 ===")
        
        # 簡易モードでDummySteadyStateが使用されることを確認
        simple_simulator = EnhancedTaxSimulator(base_model, use_simple_model=True)
        
        reform = TaxReform("テスト改革", tau_c=0.12)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = simple_simulator.simulate_reform(reform, periods=10)
            
            # DummySteadyState使用の警告確認
            dummy_warnings = [warn for warn in w 
                            if "DummySteadyState" in str(warn.message) or 
                               "HARDCODED" in str(warn.message)]
            
            assert len(dummy_warnings) > 0, "DummySteadyState使用の警告が発生していません"
            print(f"✅ DummySteadyState警告検出: {len(dummy_warnings)}個")
    
    def test_zero_division_protection(self, base_simulator):
        """ZeroDivisionError対策の検証"""
        print("\n=== ZeroDivisionError対策テスト ===")
        
        # 極端な税率でのテスト
        extreme_reforms = [
            TaxReform("ゼロ消費税", tau_c=0.0),
            TaxReform("ゼロ所得税", tau_l=0.0), 
            TaxReform("ゼロ資本税", tau_k=0.0),
            TaxReform("高消費税", tau_c=0.99),
            TaxReform("高所得税", tau_l=0.99),
        ]
        
        success_count = 0
        for reform in extreme_reforms:
            try:
                result = base_simulator.simulate_reform(reform, periods=5)
                
                # 結果の妥当性チェック
                if result and hasattr(result, 'welfare_change'):
                    if not np.isnan(result.welfare_change) and not np.isinf(result.welfare_change):
                        success_count += 1
                        print(f"✅ {reform.name}: 成功 (福利変化: {result.welfare_change:.2f}%)")
                    else:
                        print(f"⚠️ {reform.name}: 無限値/NaN検出")
                else:
                    print(f"❌ {reform.name}: 結果オブジェクト異常")
                    
            except ZeroDivisionError as e:
                pytest.fail(f"ZeroDivisionError発生: {reform.name} - {e}")
            except Exception as e:
                print(f"⚠️ {reform.name}: 計算エラー - {type(e).__name__}")
        
        assert success_count >= 2, f"極端パラメータテストの成功率が低すぎます: {success_count}/5"
    
    def test_parameter_boundary_values(self, base_model):
        """パラメータ境界値でのテスト"""
        print("\n=== パラメータ境界値テスト ===")
        
        boundary_scenarios = [
            {"name": "最小税率", "tau_c": 0.001, "tau_l": 0.001, "tau_k": 0.001, "tau_f": 0.001},
            {"name": "最大税率", "tau_c": 0.299, "tau_l": 0.499, "tau_k": 0.499, "tau_f": 0.499},
            {"name": "不均衡税率1", "tau_c": 0.50, "tau_l": 0.01, "tau_k": 0.01, "tau_f": 0.01},
            {"name": "不均衡税率2", "tau_c": 0.01, "tau_l": 0.50, "tau_k": 0.01, "tau_f": 0.01},
        ]
        
        stable_scenarios = 0
        for scenario in boundary_scenarios:
            try:
                # パラメータ設定
                params = ModelParameters.from_json('config/parameters.json')
                for key, value in scenario.items():
                    if key != "name":
                        setattr(params, key, value)
                
                # モデル作成とテスト
                test_model = DSGEModel(params)
                ss = test_model.compute_steady_state()
                
                # 定常状態の妥当性チェック
                if (ss.Y > 0 and ss.C > 0 and ss.I > 0 and ss.L > 0 and 
                    not np.isnan(ss.Y) and not np.isinf(ss.Y)):
                    stable_scenarios += 1
                    print(f"✅ {scenario['name']}: 安定定常状態 (GDP: {ss.Y:.3f})")
                else:
                    print(f"⚠️ {scenario['name']}: 不安定定常状態")
                    
            except Exception as e:
                print(f"❌ {scenario['name']}: エラー - {type(e).__name__}")
        
        assert stable_scenarios >= 2, f"境界値テストの安定性が不十分: {stable_scenarios}/4"
    
    def test_api_consistency_different_modes(self, base_model):
        """API一貫性: 異なるモード間での互換性"""
        print("\n=== API一貫性テスト ===")
        
        reform = TaxReform("一貫性テスト", tau_c=0.12, tau_l=0.18)
        
        # 異なるモードでのシミュレータ作成
        simulators = {
            "完全モード": EnhancedTaxSimulator(base_model, use_simple_model=False),
            "簡易モード": EnhancedTaxSimulator(base_model, use_simple_model=True),
            "線形化簡易": EnhancedTaxSimulator(base_model, use_simple_linearization=True),
        }
        
        results = {}
        api_consistent = 0
        
        for mode_name, simulator in simulators.items():
            try:
                result = simulator.simulate_reform(reform, periods=10)
                
                # 基本APIの存在確認
                required_attrs = ['name', 'welfare_change', 'baseline_path', 'reform_path']
                missing_attrs = [attr for attr in required_attrs if not hasattr(result, attr)]
                
                if not missing_attrs:
                    api_consistent += 1
                    results[mode_name] = result
                    print(f"✅ {mode_name}: API一貫性確認")
                else:
                    print(f"❌ {mode_name}: 不足属性 {missing_attrs}")
                    
            except Exception as e:
                print(f"❌ {mode_name}: 実行エラー - {type(e).__name__}")
        
        assert api_consistent >= 2, f"API一貫性が不十分: {api_consistent}/3モード"
    
    def test_memory_efficiency_long_simulation(self, base_simulator):
        """メモリ効率性: 長期シミュレーションでの安定性"""
        print("\n=== メモリ効率性テスト ===")
        
        import psutil
        import gc
        
        # メモリ使用量の初期値
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        reforms = [
            TaxReform(f"テスト改革{i}", tau_c=0.10 + i*0.01, tau_l=0.20 - i*0.005)
            for i in range(10)
        ]
        
        memory_stable = True
        for i, reform in enumerate(reforms):
            try:
                # 長期シミュレーション実行
                result = base_simulator.simulate_reform(reform, periods=80)
                
                # メモリチェック
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                
                if memory_increase > 500:  # 500MB以上の増加は異常
                    print(f"⚠️ 改革{i}: メモリ使用量異常 (+{memory_increase:.1f}MB)")
                    memory_stable = False
                
                # ガベージコレクション
                gc.collect()
                
            except Exception as e:
                print(f"❌ 改革{i}: エラー - {type(e).__name__}")
                memory_stable = False
        
        final_memory = process.memory_info().rss / 1024 / 1024
        total_increase = final_memory - initial_memory
        
        print(f"メモリ使用量: {initial_memory:.1f}MB → {final_memory:.1f}MB (+{total_increase:.1f}MB)")
        
        assert memory_stable, "メモリ使用量が異常に増加しています"
        assert total_increase < 200, f"メモリリークの可能性: +{total_increase:.1f}MB"
    
    def test_error_recovery_mechanisms(self, base_model):
        """エラー回復メカニズムのテスト"""
        print("\n=== エラー回復メカニズムテスト ===")
        
        # 意図的に失敗を誘発するシナリオ
        problematic_scenarios = [
            {"name": "不正税率組み合わせ", "tau_c": 1.5, "tau_l": 1.2},  # 100%超
            {"name": "負の税率", "tau_c": -0.1, "tau_l": -0.05},
            {"name": "極端な不均衡", "tau_c": 0.99, "tau_l": 0.99, "tau_k": 0.99, "tau_f": 0.99},
        ]
        
        recovery_success = 0
        for scenario in problematic_scenarios:
            try:
                # 不正パラメータでシミュレータ作成を試行
                params = ModelParameters.from_json('config/parameters.json')
                
                for key, value in scenario.items():
                    if key != "name":
                        setattr(params, key, value)
                
                # エラーハンドリングの確認
                try:
                    test_model = DSGEModel(params)
                    simulator = EnhancedTaxSimulator(test_model)
                    print(f"⚠️ {scenario['name']}: 予期しない成功")
                except Exception as expected_error:
                    # 適切なエラーハンドリングが発生
                    if not isinstance(expected_error, (ValueError, RuntimeError, OverflowError)):
                        print(f"❌ {scenario['name']}: 予期しないエラー型 - {type(expected_error).__name__}")
                    else:
                        recovery_success += 1
                        print(f"✅ {scenario['name']}: 適切なエラーハンドリング - {type(expected_error).__name__}")
                
            except Exception as e:
                print(f"❌ {scenario['name']}: テスト実行エラー - {type(e).__name__}")
        
        assert recovery_success >= 2, f"エラー回復メカニズムが不十分: {recovery_success}/3"
    
    def test_concurrent_simulation_safety(self, base_model):
        """並行シミュレーションの安全性"""
        print("\n=== 並行シミュレーション安全性テスト ===")
        
        import threading
        import queue
        
        def run_simulation(simulator, reform, result_queue):
            """スレッド用シミュレーション実行関数"""
            try:
                result = simulator.simulate_reform(reform, periods=20)
                result_queue.put(("success", reform.name, result))
            except Exception as e:
                result_queue.put(("error", reform.name, str(e)))
        
        # 複数のシミュレータとリフォーム
        reforms = [
            TaxReform(f"並行テスト{i}", tau_c=0.10 + i*0.01)
            for i in range(5)
        ]
        
        simulators = [EnhancedTaxSimulator(base_model) for _ in range(3)]
        
        threads = []
        result_queue = queue.Queue()
        
        # 並行実行
        for i, (simulator, reform) in enumerate(zip(simulators, reforms)):
            thread = threading.Thread(
                target=run_simulation,
                args=(simulator, reform, result_queue)
            )
            threads.append(thread)
            thread.start()
        
        # スレッド終了を待機
        for thread in threads:
            thread.join(timeout=30)  # 30秒タイムアウト
        
        # 結果収集
        successful_runs = 0
        while not result_queue.empty():
            status, name, result = result_queue.get()
            if status == "success":
                successful_runs += 1
                print(f"✅ {name}: 並行実行成功")
            else:
                print(f"❌ {name}: 並行実行エラー - {result}")
        
        assert successful_runs >= 3, f"並行実行の成功率が低すぎます: {successful_runs}/5"
    
    def test_notebook_cell_isolation(self, base_model):
        """Notebookセル分離のテスト（状態汚染防止）"""
        print("\n=== Notebookセル分離テスト ===")
        
        # 最初のセル: シミュレータ作成
        simulator1 = EnhancedTaxSimulator(base_model)
        reform1 = TaxReform("セル1改革", tau_c=0.12)
        result1 = simulator1.simulate_reform(reform1, periods=10)
        
        # 状態確認
        initial_state = {
            'welfare_change': result1.welfare_change,
            'simulator_type': type(simulator1).__name__
        }
        
        # 2番目のセル: 異なる設定
        simulator2 = EnhancedTaxSimulator(base_model, use_simple_model=True)
        reform2 = TaxReform("セル2改革", tau_l=0.15)
        result2 = simulator2.simulate_reform(reform2, periods=10)
        
        # 3番目のセル: 最初のシミュレータ再利用
        reform3 = TaxReform("セル3改革", tau_c=0.11)
        result3 = simulator1.simulate_reform(reform3, periods=10)
        
        # 状態汚染チェック
        state_preserved = True
        
        # シミュレータ1の状態が保持されているか
        if hasattr(simulator1, 'use_simple_model'):
            if simulator1.use_simple_model != False:
                state_preserved = False
                print("❌ シミュレータ1の設定が変更されています")
        
        # 結果オブジェクトの独立性
        if result1.name == result3.name:
            state_preserved = False
            print("❌ 結果オブジェクトが共有されています")
        
        if state_preserved:
            print("✅ セル間の状態分離が適切に機能")
        
        assert state_preserved, "Notebookセル間の状態汚染が発生しています"


def main():
    """テストスイートの実行"""
    print("=" * 60)
    print("拡張Notebookテストスイート実行開始")
    print("=" * 60)
    
    # pytest実行
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--capture=no"
    ])
    
    print("\n" + "=" * 60)
    print(f"テスト完了 - 終了コード: {exit_code}")
    print("=" * 60)
    
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)