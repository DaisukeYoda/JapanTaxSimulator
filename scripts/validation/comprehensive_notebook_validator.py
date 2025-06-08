#!/usr/bin/env python3
"""
包括的Notebook検証スクリプト

全notebookの動作を検証し、研究整合性、エラーハンドリング、
パフォーマンスを総合的にテストします。
"""

import os
import sys
import time
import json
import subprocess
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

# プロジェクトルートを設定
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

try:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError
    NBCLIENT_AVAILABLE = True
except ImportError:
    print("⚠️ nbclientが利用できません。基本的な検証のみ実行します。")
    NBCLIENT_AVAILABLE = False

from src.dsge_model import DSGEModel, ModelParameters
from src.tax_simulator import EnhancedTaxSimulator, TaxReform
from src.research_warnings import ResearchWarning


class NotebookValidator:
    """包括的Notebook検証クラス"""
    
    def __init__(self):
        self.results = {}
        self.notebooks_dir = project_root / "notebooks"
        self.test_report = {
            'summary': {},
            'detailed_results': {},
            'research_integrity': {},
            'performance_metrics': {},
            'error_scenarios': {}
        }
    
    def validate_all_notebooks(self) -> Dict[str, Any]:
        """全notebookの包括的検証"""
        print("=" * 60)
        print("包括的Notebook検証開始")
        print("=" * 60)
        
        notebook_files = list(self.notebooks_dir.glob("*.ipynb"))
        
        for nb_file in notebook_files:
            print(f"\n{'='*50}")
            print(f"検証中: {nb_file.name}")
            print(f"{'='*50}")
            
            result = self.validate_single_notebook(nb_file)
            self.results[str(nb_file)] = result
        
        # 検証レポートの生成
        self.generate_comprehensive_report()
        
        return self.results
    
    def validate_single_notebook(self, notebook_path: Path) -> Dict[str, Any]:
        """単一notebookの検証"""
        result = {
            'basic_execution': self._test_basic_execution(notebook_path),
            'research_integrity': self._test_research_integrity(notebook_path),
            'error_handling': self._test_error_handling(notebook_path),
            'performance': self._test_performance(notebook_path),
            'api_compatibility': self._test_api_compatibility(notebook_path)
        }
        
        return result
    
    def _test_basic_execution(self, notebook_path: Path) -> Dict[str, Any]:
        """基本的な実行テスト"""
        print("\n--- 基本実行テスト ---")
        
        if not NBCLIENT_AVAILABLE:
            return {'status': 'skipped', 'reason': 'nbclient unavailable'}
        
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            # 基本的な初期化コードを挿入
            self._inject_test_setup(nb)
            
            client = NotebookClient(
                nb,
                timeout=300,
                kernel_name='python3',
                allow_errors=True
            )
            
            start_time = time.time()
            
            try:
                client.execute()
                execution_time = time.time() - start_time
                
                # セル実行結果の分析
                executed_cells = 0
                error_cells = 0
                
                for cell in nb.cells:
                    if cell.cell_type == 'code' and cell.get('outputs'):
                        executed_cells += 1
                        
                        for output in cell.outputs:
                            if output.get('output_type') == 'error':
                                error_cells += 1
                                break
                
                success_rate = (executed_cells - error_cells) / max(executed_cells, 1)
                
                print(f"✅ 実行完了: {executed_cells}セル実行、{error_cells}エラー")
                print(f"成功率: {success_rate:.1%}、実行時間: {execution_time:.1f}秒")
                
                return {
                    'status': 'success',
                    'executed_cells': executed_cells,
                    'error_cells': error_cells,
                    'success_rate': success_rate,
                    'execution_time': execution_time
                }
                
            except Exception as e:
                print(f"❌ 実行エラー: {e}")
                return {
                    'status': 'execution_failed',
                    'error': str(e),
                    'execution_time': time.time() - start_time
                }
                
        except Exception as e:
            print(f"❌ ノートブック読み込みエラー: {e}")
            return {'status': 'load_failed', 'error': str(e)}
    
    def _test_research_integrity(self, notebook_path: Path) -> Dict[str, Any]:
        """研究整合性テスト"""
        print("\n--- 研究整合性テスト ---")
        
        integrity_issues = []
        warnings_detected = []
        
        try:
            # モデル初期化テスト
            params = ModelParameters.from_json('config/parameters.json')
            model = DSGEModel(params)
            
            # DummySteadyState検出テスト
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                # 簡易モードでのテスト
                simple_simulator = EnhancedTaxSimulator(model, use_simple_model=True)
                reform = TaxReform("整合性テスト", tau_c=0.12)
                result = simple_simulator.simulate_reform(reform, periods=5)
                
                # 警告の分析
                for warning in w:
                    warning_msg = str(warning.message)
                    if any(keyword in warning_msg.lower() for keyword in 
                          ['dummy', 'hardcoded', 'research', 'fallback']):
                        warnings_detected.append(warning_msg)
                        
                        if 'dummy' in warning_msg.lower():
                            integrity_issues.append("DummySteadyState使用検出")
            
            # DummySteadyStateインスタンス検出（クラス名で判定）
            if hasattr(result, 'steady_state_baseline'):
                baseline_type = type(result.steady_state_baseline).__name__
                if 'Dummy' in baseline_type:
                    integrity_issues.append(f"DummySteadyStateインスタンス使用: {baseline_type}")
            
            if hasattr(result, 'steady_state_reform'):
                reform_type = type(result.steady_state_reform).__name__
                if 'Dummy' in reform_type:
                    integrity_issues.append(f"DummySteadyStateインスタンス使用: {reform_type}")
            
            # 福利厚生計算の検証
            if hasattr(result, 'welfare_change'):
                if result.welfare_change == 0.0:
                    integrity_issues.append("福利変化ゼロ（計算失敗の可能性）")
                elif np.isnan(result.welfare_change) or np.isinf(result.welfare_change):
                    integrity_issues.append("福利変化が無限値/NaN")
            
            print(f"警告検出: {len(warnings_detected)}個")
            print(f"整合性問題: {len(integrity_issues)}個")
            
            if integrity_issues:
                for issue in integrity_issues:
                    print(f"  ⚠️ {issue}")
            
            status = 'clean' if not integrity_issues else 'issues_found'
            
            return {
                'status': status,
                'integrity_issues': integrity_issues,
                'warnings_detected': warnings_detected,
                'dummy_state_used': 'DummySteadyState' in str(integrity_issues)
            }
            
        except Exception as e:
            print(f"❌ 研究整合性テストエラー: {e}")
            return {
                'status': 'test_failed',
                'error': str(e)
            }
    
    def _test_error_handling(self, notebook_path: Path) -> Dict[str, Any]:
        """エラーハンドリングテスト"""
        print("\n--- エラーハンドリングテスト ---")
        
        error_scenarios = [
            {'name': '極端な税率', 'params': {'tau_c': 0.99, 'tau_l': 0.99}},
            {'name': 'ゼロ税率', 'params': {'tau_c': 0.0, 'tau_l': 0.0}},
            {'name': '負の税率', 'params': {'tau_c': -0.1, 'tau_l': -0.1}},
        ]
        
        handled_errors = 0
        total_scenarios = len(error_scenarios)
        
        for scenario in error_scenarios:
            try:
                # 基本モデル作成
                params = ModelParameters.from_json('config/parameters.json')
                model = DSGEModel(params)
                simulator = EnhancedTaxSimulator(model)
                
                # エラーシナリオ実行
                reform = TaxReform(
                    scenario['name'],
                    **scenario['params']
                )
                
                try:
                    result = simulator.simulate_reform(reform, periods=5)
                    
                    # 結果の妥当性チェック
                    if (hasattr(result, 'welfare_change') and 
                        (np.isnan(result.welfare_change) or np.isinf(result.welfare_change))):
                        handled_errors += 1
                        print(f"✅ {scenario['name']}: 異常結果検出")
                    else:
                        print(f"⚠️ {scenario['name']}: 予期しない正常終了")
                        
                except Exception as expected_error:
                    handled_errors += 1
                    print(f"✅ {scenario['name']}: 適切なエラー - {type(expected_error).__name__}")
                    
            except Exception as e:
                print(f"❌ {scenario['name']}: テスト実行エラー - {type(e).__name__}")
        
        error_handling_rate = handled_errors / total_scenarios
        
        print(f"エラーハンドリング成功率: {error_handling_rate:.1%}")
        
        return {
            'status': 'good' if error_handling_rate >= 0.7 else 'needs_improvement',
            'handling_rate': error_handling_rate,
            'scenarios_tested': total_scenarios,
            'scenarios_handled': handled_errors
        }
    
    def _test_performance(self, notebook_path: Path) -> Dict[str, Any]:
        """パフォーマンステスト"""
        print("\n--- パフォーマンステスト ---")
        
        try:
            params = ModelParameters.from_json('config/parameters.json')
            model = DSGEModel(params)
            
            # 異なるモードでのパフォーマンス比較
            performance_data = {}
            
            test_scenarios = [
                {'name': '簡易モード', 'config': {'use_simple_model': True}},
                {'name': '完全モード', 'config': {'use_simple_model': False}},
                {'name': '簡易線形化', 'config': {'use_simple_linearization': True}}
            ]
            
            for scenario in test_scenarios:
                try:
                    start_time = time.time()
                    
                    simulator = EnhancedTaxSimulator(model, **scenario['config'])
                    reform = TaxReform("パフォーマンステスト", tau_c=0.12)
                    result = simulator.simulate_reform(reform, periods=20)
                    
                    execution_time = time.time() - start_time
                    
                    performance_data[scenario['name']] = {
                        'execution_time': execution_time,
                        'success': True
                    }
                    
                    print(f"✅ {scenario['name']}: {execution_time:.2f}秒")
                    
                except Exception as e:
                    performance_data[scenario['name']] = {
                        'execution_time': None,
                        'success': False,
                        'error': str(e)
                    }
                    print(f"❌ {scenario['name']}: エラー - {type(e).__name__}")
            
            return {
                'status': 'completed',
                'performance_data': performance_data
            }
            
        except Exception as e:
            print(f"❌ パフォーマンステストエラー: {e}")
            return {
                'status': 'test_failed',
                'error': str(e)
            }
    
    def _test_api_compatibility(self, notebook_path: Path) -> Dict[str, Any]:
        """API互換性テスト"""
        print("\n--- API互換性テスト ---")
        
        try:
            params = ModelParameters.from_json('config/parameters.json')
            model = DSGEModel(params)
            
            # 異なる設定でのAPI一貫性確認
            api_tests = [
                {'name': 'デフォルト設定', 'args': [], 'kwargs': {}},
                {'name': '簡易モード', 'args': [], 'kwargs': {'use_simple_model': True}},
                {'name': '簡易線形化', 'args': [], 'kwargs': {'use_simple_linearization': True}}
            ]
            
            api_compatibility = {}
            required_attributes = ['name', 'welfare_change', 'baseline_path', 'reform_path']
            
            for test_config in api_tests:
                try:
                    simulator = EnhancedTaxSimulator(model, *test_config['args'], **test_config['kwargs'])
                    reform = TaxReform("API互換性テスト", tau_c=0.11)
                    result = simulator.simulate_reform(reform, periods=10)
                    
                    # API一貫性チェック
                    missing_attrs = []
                    for attr in required_attributes:
                        if not hasattr(result, attr):
                            missing_attrs.append(attr)
                    
                    api_compatibility[test_config['name']] = {
                        'success': len(missing_attrs) == 0,
                        'missing_attributes': missing_attrs
                    }
                    
                    if missing_attrs:
                        print(f"⚠️ {test_config['name']}: 不足属性 {missing_attrs}")
                    else:
                        print(f"✅ {test_config['name']}: API互換性確認")
                        
                except Exception as e:
                    api_compatibility[test_config['name']] = {
                        'success': False,
                        'error': str(e)
                    }
                    print(f"❌ {test_config['name']}: エラー - {type(e).__name__}")
            
            compatible_apis = sum(1 for test in api_compatibility.values() if test.get('success', False))
            compatibility_rate = compatible_apis / len(api_tests)
            
            return {
                'status': 'good' if compatibility_rate >= 0.8 else 'needs_improvement',
                'compatibility_rate': compatibility_rate,
                'detailed_results': api_compatibility
            }
            
        except Exception as e:
            print(f"❌ API互換性テストエラー: {e}")
            return {
                'status': 'test_failed',
                'error': str(e)
            }
    
    def _inject_test_setup(self, nb):
        """テスト用の初期化コードを挿入"""
        setup_code = '''
# テスト実行用設定
import os
import sys
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# プロジェクトルート設定
project_root = os.path.abspath('.')
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# resultsディレクトリ作成
os.makedirs('results', exist_ok=True)
'''
        
        setup_cell = nbformat.v4.new_code_cell(source=setup_code)
        nb.cells.insert(0, setup_cell)
    
    def generate_comprehensive_report(self):
        """包括的レポートの生成"""
        print("\n" + "=" * 60)
        print("包括的検証レポート生成中...")
        print("=" * 60)
        
        # サマリー統計の計算
        total_notebooks = len(self.results)
        successful_executions = sum(1 for r in self.results.values() 
                                  if r.get('basic_execution', {}).get('status') == 'success')
        
        research_clean = sum(1 for r in self.results.values()
                           if r.get('research_integrity', {}).get('status') == 'clean')
        
        good_error_handling = sum(1 for r in self.results.values()
                                if r.get('error_handling', {}).get('status') == 'good')
        
        # レポート作成
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total_notebooks': total_notebooks,
                'successful_executions': successful_executions,
                'execution_success_rate': successful_executions / max(total_notebooks, 1),
                'research_integrity_clean': research_clean,
                'good_error_handling': good_error_handling
            },
            'detailed_results': self.results,
            'recommendations': self._generate_recommendations()
        }
        
        # レポートファイル保存
        report_file = project_root / 'results' / 'comprehensive_notebook_validation_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # 人間可読レポート
        text_report_file = project_root / 'results' / 'notebook_validation_summary.txt'
        with open(text_report_file, 'w', encoding='utf-8') as f:
            self._write_text_report(f, report)
        
        print(f"\n📊 レポート保存先:")
        print(f"  詳細: {report_file}")
        print(f"  サマリー: {text_report_file}")
        
        # コンソール出力
        self._print_summary(report)
    
    def _generate_recommendations(self) -> List[str]:
        """改善提案の生成"""
        recommendations = []
        
        # 研究整合性の問題
        dummy_usage_detected = any(
            r.get('research_integrity', {}).get('dummy_state_used', False)
            for r in self.results.values()
        )
        
        if dummy_usage_detected:
            recommendations.append(
                "🚨 CRITICAL: DummySteadyState使用検出 - 学術研究には使用不可。"
                "use_simple_model=Falseを使用してください。"
            )
        
        # エラーハンドリング
        poor_error_handling = any(
            r.get('error_handling', {}).get('status') == 'needs_improvement'
            for r in self.results.values()
        )
        
        if poor_error_handling:
            recommendations.append(
                "⚠️ エラーハンドリングの改善が必要。極端なパラメータでの"
                "robust性を向上させてください。"
            )
        
        return recommendations
    
    def _write_text_report(self, f, report):
        """テキストレポートの書き出し"""
        f.write("Notebook包括的検証レポート\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"検証日時: {report['timestamp']}\n")
        f.write(f"対象Notebook数: {report['summary']['total_notebooks']}\n\n")
        
        f.write("実行成功率: {:.1%}\n".format(report['summary']['execution_success_rate']))
        f.write(f"研究整合性クリーン: {report['summary']['research_integrity_clean']}件\n")
        f.write(f"エラーハンドリング良好: {report['summary']['good_error_handling']}件\n\n")
        
        f.write("改善提案:\n")
        for rec in report['recommendations']:
            f.write(f"  - {rec}\n")
        
        f.write("\n詳細結果:\n")
        for notebook, result in report['detailed_results'].items():
            f.write(f"\n{Path(notebook).name}:\n")
            f.write(f"  基本実行: {result.get('basic_execution', {}).get('status', 'unknown')}\n")
            f.write(f"  研究整合性: {result.get('research_integrity', {}).get('status', 'unknown')}\n")
            f.write(f"  エラーハンドリング: {result.get('error_handling', {}).get('status', 'unknown')}\n")
    
    def _print_summary(self, report):
        """サマリーをコンソールに表示"""
        print("\n" + "=" * 50)
        print("📋 検証結果サマリー")
        print("=" * 50)
        
        summary = report['summary']
        print(f"対象Notebook数: {summary['total_notebooks']}")
        print(f"実行成功率: {summary['execution_success_rate']:.1%}")
        print(f"研究整合性クリーン: {summary['research_integrity_clean']}/{summary['total_notebooks']}")
        print(f"エラーハンドリング良好: {summary['good_error_handling']}/{summary['total_notebooks']}")
        
        if report['recommendations']:
            print("\n🔧 主な改善提案:")
            for rec in report['recommendations'][:3]:  # 上位3つ
                print(f"  • {rec}")
        
        print("\n✅ 検証完了")


def main():
    """メイン実行関数"""
    validator = NotebookValidator()
    results = validator.validate_all_notebooks()
    
    return 0 if results else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)