"""
線形化手法オプションのテストスクリプト

新しい use_simple_linearization パラメータの動作を確認します。
"""

import sys
import os
from pathlib import Path

# Set up imports
project_root = Path(__file__).parent.parent.parent
os.chdir(str(project_root))
sys.path.insert(0, str(project_root))

from src.dsge_model import DSGEModel, ModelParameters
from src.tax_simulator import EnhancedTaxSimulator, TaxReform

def test_linearization_options():
    """線形化オプションのテスト"""
    print("🧪 線形化手法オプションのテスト開始\n")
    
    # ベースラインモデルの構築
    params = ModelParameters.from_json("config/parameters.json")
    model = DSGEModel(params)
    baseline_ss = model.compute_steady_state()
    
    if baseline_ss is None:
        print("❌ ベースライン定常状態の計算に失敗")
        return
    
    # テスト用の税制改革
    test_reform = TaxReform(
        name="テスト用消費税増税_2pp",
        tau_c=params.tau_c + 0.02,
        implementation='permanent'
    )
    
    print("=" * 60)
    print("1️⃣ 簡略化線形化を明示的に指定（デモ用途）")
    print("=" * 60)
    try:
        simulator1 = EnhancedTaxSimulator(
            model, 
            use_simple_model=False,
            use_simple_linearization=True
        )
        print("✅ 簡略化線形化での初期化完了\n")
    except Exception as e:
        print(f"❌ エラー: {e}\n")
    
    print("=" * 60)
    print("2️⃣ 完全線形化を明示的に指定（学術研究用途）")
    print("=" * 60)
    try:
        simulator2 = EnhancedTaxSimulator(
            model,
            use_simple_model=False, 
            use_simple_linearization=False
        )
        print("✅ 完全線形化での初期化完了\n")
    except Exception as e:
        print(f"❌ エラー: {e}\n")
    
    print("=" * 60)
    print("3️⃣ 自動選択（デフォルト動作、警告付き）")
    print("=" * 60)
    try:
        simulator3 = EnhancedTaxSimulator(
            model,
            use_simple_model=False,
            use_simple_linearization=None  # 明示的にNoneを指定
        )
        print("✅ 自動選択での初期化完了\n")
    except Exception as e:
        print(f"❌ エラー: {e}\n")
    
    print("=" * 60)
    print("4️⃣ 旧来の動作（互換性確認）")
    print("=" * 60)
    try:
        # 従来通りのパラメータで作成
        simulator4 = EnhancedTaxSimulator(model, use_simple_model=False)
        print("✅ 旧来動作での初期化完了\n")
    except Exception as e:
        print(f"❌ エラー: {e}\n")
    
    print("🎉 全テスト完了！")
    print("\n📋 使用推奨事項:")
    print("  🎓 学術研究・政策分析: use_simple_linearization=False")
    print("  📚 デモ・教育用途: use_simple_linearization=True")
    print("  ⚠️  自動選択は推奨しません（明示的な指定が重要）")

if __name__ == "__main__":
    test_linearization_options()