#!/usr/bin/env python3
"""
Jupyter Notebook自動テストスクリプト

使用方法:
    python scripts/test_notebooks.py
    python scripts/test_notebooks.py --notebook notebooks/tax_simulation_demo.ipynb
    python scripts/test_notebooks.py --fix-errors  # エラー自動修正を試行
"""

import os
import sys
import argparse
import subprocess
import json
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


class NotebookTester:
    """Jupyter Notebookの実行とテストを行うクラス"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.results = {}
        
    def test_all_notebooks(self) -> Dict[str, Any]:
        """全てのnotebookをテスト"""
        notebooks_dir = self.project_root / "notebooks"
        notebook_files = list(notebooks_dir.glob("*.ipynb"))
        
        print(f"Found {len(notebook_files)} notebooks to test:")
        for nb_file in notebook_files:
            print(f"  - {nb_file.name}")
        
        for nb_file in notebook_files:
            print(f"\n{'='*60}")
            print(f"Testing: {nb_file.name}")
            print(f"{'='*60}")
            
            result = self.test_notebook(nb_file)
            self.results[str(nb_file)] = result
            
        return self.results
    
    def test_notebook(self, notebook_path: Path) -> Dict[str, Any]:
        """単一notebookのテスト"""
        result = {
            'success': False,
            'total_cells': 0,
            'executed_cells': 0,
            'failed_cells': [],
            'errors': [],
            'execution_time': 0
        }
        
        try:
            # Notebookを読み込み
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            # 共通の初期化コードを先頭に挿入
            self._inject_common_setup(nb)
            
            result['total_cells'] = len([cell for cell in nb.cells if cell.cell_type == 'code'])
            
            # NotebookClientで実行
            client = NotebookClient(
                nb,
                timeout=300,  # 5分タイムアウト
                kernel_name='python3',
                allow_errors=True,  # エラーがあっても続行
                resources={'metadata': {'path': str(notebook_path.parent)}}  # 実行ディレクトリを設定
            )
            
            # 実行
            import time
            start_time = time.time()
            
            try:
                client.execute()
                result['success'] = True
            except Exception as e:
                print(f"Notebook execution failed: {e}")
                result['errors'].append(str(e))
            
            result['execution_time'] = time.time() - start_time
            
            # セル毎の結果を分析
            for i, cell in enumerate(nb.cells):
                if cell.cell_type == 'code':
                    if cell.get('outputs'):
                        # 出力があれば実行成功
                        result['executed_cells'] += 1
                        
                        # エラー出力をチェック
                        for output in cell.outputs:
                            if output.get('output_type') == 'error':
                                error_info = {
                                    'cell_index': i,
                                    'error_name': output.get('ename', 'Unknown'),
                                    'error_value': output.get('evalue', ''),
                                    'traceback': output.get('traceback', [])
                                }
                                result['failed_cells'].append(error_info)
                                result['errors'].append(f"Cell {i}: {error_info['error_name']}: {error_info['error_value']}")
            
            self._print_result(notebook_path.name, result)
            return result
            
        except Exception as e:
            print(f"Failed to test notebook {notebook_path}: {e}")
            result['errors'].append(f"Failed to load/execute notebook: {e}")
            return result
    
    def _print_result(self, notebook_name: str, result: Dict[str, Any]):
        """結果を出力"""
        print(f"\nResults for {notebook_name}:")
        print(f"  Total code cells: {result['total_cells']}")
        print(f"  Successfully executed: {result['executed_cells']}")
        print(f"  Failed cells: {len(result['failed_cells'])}")
        print(f"  Execution time: {result['execution_time']:.1f}s")
        
        if result['failed_cells']:
            print("\n  Errors:")
            for error in result['errors']:
                print(f"    - {error}")
        
        status = "✅ PASSED" if result['success'] and not result['failed_cells'] else "❌ FAILED"
        print(f"  Status: {status}")
    
    def generate_test_report(self, output_file: str = None):
        """テスト結果レポートを生成"""
        if not output_file:
            output_file = self.project_root / "results" / "notebook_test_report.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Jupyter Notebook Test Report\n")
            f.write("=" * 50 + "\n\n")
            
            total_notebooks = len(self.results)
            passed_notebooks = sum(1 for r in self.results.values() 
                                 if r['success'] and not r['failed_cells'])
            
            f.write(f"Summary:\n")
            f.write(f"  Total notebooks: {total_notebooks}\n")
            f.write(f"  Passed: {passed_notebooks}\n")
            f.write(f"  Failed: {total_notebooks - passed_notebooks}\n\n")
            
            for notebook_path, result in self.results.items():
                f.write(f"Notebook: {Path(notebook_path).name}\n")
                f.write(f"  Total cells: {result['total_cells']}\n")
                f.write(f"  Executed: {result['executed_cells']}\n")
                f.write(f"  Failed: {len(result['failed_cells'])}\n")
                f.write(f"  Time: {result['execution_time']:.1f}s\n")
                
                if result['errors']:
                    f.write("  Errors:\n")
                    for error in result['errors']:
                        f.write(f"    - {error}\n")
                f.write("\n")
        
        print(f"\nTest report saved to: {output_file}")
    
    def _inject_common_setup(self, nb):
        """Notebookに共通の初期化コードを挿入"""
        setup_code = '''
# テスト実行用の共通設定
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# プロジェクトルートをパスに追加
project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 作業ディレクトリをプロジェクトルートに設定
os.chdir(project_root)

# resultsディレクトリを作成
os.makedirs('results', exist_ok=True)
'''
        
        # 最初のコードセルの前に挿入
        setup_cell = new_code_cell(source=setup_code)
        nb.cells.insert(0, setup_cell)


class NotebookFixer:
    """Notebookの一般的なエラーを自動修正するクラス"""
    
    def __init__(self):
        self.common_fixes = {
            "missing_steady_state_arg": {
                "pattern": "ImprovedLinearizedDSGE(model)",
                "replacement": "ImprovedLinearizedDSGE(model, steady_state)"
            },
            "wrong_shock_arg": {
                "pattern": "shock=",
                "replacement": "shock_type="
            },
            "missing_import": {
                "pattern": "from src.linearization import",
                "replacement": "from src.linearization_improved import"
            }
        }
    
    def fix_notebook(self, notebook_path: Path) -> bool:
        """Notebookの一般的なエラーを修正"""
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            modified = False
            for cell in nb.cells:
                if cell.cell_type == 'code':
                    original_source = cell.source
                    
                    # 一般的な修正を適用
                    for fix_name, fix_data in self.common_fixes.items():
                        if fix_data["pattern"] in cell.source:
                            cell.source = cell.source.replace(
                                fix_data["pattern"], 
                                fix_data["replacement"]
                            )
                            modified = True
                            print(f"Applied fix '{fix_name}' to cell")
            
            if modified:
                # バックアップを作成
                backup_path = notebook_path.with_suffix('.ipynb.backup')
                notebook_path.rename(backup_path)
                print(f"Created backup: {backup_path}")
                
                # 修正版を保存
                with open(notebook_path, 'w', encoding='utf-8') as f:
                    nbformat.write(nb, f)
                print(f"Fixed notebook saved: {notebook_path}")
                
            return modified
            
        except Exception as e:
            print(f"Failed to fix notebook {notebook_path}: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Test Jupyter Notebooks")
    parser.add_argument("--notebook", type=str, help="Test specific notebook")
    parser.add_argument("--fix-errors", action="store_true", help="Try to fix common errors")
    parser.add_argument("--output", type=str, help="Output file for test report")
    
    args = parser.parse_args()
    
    # プロジェクトルートに移動
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    tester = NotebookTester(project_root)
    
    if args.fix_errors:
        print("🔧 Attempting to fix common notebook errors...")
        fixer = NotebookFixer()
        
        notebooks_dir = project_root / "notebooks"
        for nb_file in notebooks_dir.glob("*.ipynb"):
            print(f"\nChecking {nb_file.name}...")
            if fixer.fix_notebook(nb_file):
                print(f"✅ Fixed {nb_file.name}")
            else:
                print(f"ℹ️  No fixes needed for {nb_file.name}")
    
    if args.notebook:
        # 特定のnotebookをテスト
        notebook_path = Path(args.notebook)
        if not notebook_path.exists():
            print(f"Notebook not found: {notebook_path}")
            sys.exit(1)
        
        result = tester.test_notebook(notebook_path)
    else:
        # 全てのnotebookをテスト
        print("🧪 Testing all notebooks...")
        results = tester.test_all_notebooks()
    
    # レポート生成
    tester.generate_test_report(args.output)
    
    # 結果サマリー
    total = len(tester.results)
    passed = sum(1 for r in tester.results.values() if r['success'] and not r['failed_cells'])
    
    print(f"\n{'='*60}")
    print(f"Test Summary: {passed}/{total} notebooks passed")
    
    if passed == total:
        print("🎉 All notebooks are working!")
        sys.exit(0)
    else:
        print("❌ Some notebooks have issues. Check the report for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()