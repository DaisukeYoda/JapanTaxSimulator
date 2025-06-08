"""
テストファイルの一括修正スクリプト
新しいuse_simple_linearizationパラメータに対応
"""

import re
from pathlib import Path

def fix_test_file(file_path):
    """テストファイルを修正"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # EnhancedTaxSimulator(model) のパターンを修正
    # ただし、すでに修正済みのものは除外
    pattern1 = r'EnhancedTaxSimulator\(model\)(?!\s*#.*already.*fixed)'
    replacement1 = 'EnhancedTaxSimulator(model, use_simple_model=False, use_simple_linearization=True)'
    content = re.sub(pattern1, replacement1, content)
    
    # EnhancedTaxSimulator(baseline_model) のパターンも修正
    pattern2 = r'EnhancedTaxSimulator\(baseline_model\)(?!\s*#.*already.*fixed)'
    replacement2 = 'EnhancedTaxSimulator(baseline_model, use_simple_model=False, use_simple_linearization=True)'
    content = re.sub(pattern2, replacement2, content)
    
    # その他の単一引数パターン
    pattern3 = r'EnhancedTaxSimulator\(([^,)]+)\)(?!\s*#.*already.*fixed)'
    def replace_func(match):
        arg = match.group(1).strip()
        if 'use_simple_model' in arg or 'use_simple_linearization' in arg:
            return match.group(0)  # すでに修正済み
        return f'EnhancedTaxSimulator({arg}, use_simple_model=False, use_simple_linearization=True)'
    
    content = re.sub(pattern3, replace_func, content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ 修正完了: {file_path}")

def main():
    """メイン実行"""
    test_file = Path('tests/unit/test_simulation_engine.py')
    
    if test_file.exists():
        print(f"🔧 テストファイルを修正中: {test_file}")
        fix_test_file(test_file)
    else:
        print(f"❌ ファイルが見つかりません: {test_file}")

if __name__ == "__main__":
    main()