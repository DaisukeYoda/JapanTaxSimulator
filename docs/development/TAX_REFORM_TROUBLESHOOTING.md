# 税制改革シミュレーション トラブルシューティングガイド

## 概要

本ガイドは、Japan Tax Simulatorで税制改革シミュレーションを実行する際に発生する可能性のある問題とその解決方法を説明します。Issue #6の解決経験を基に、実用的なトラブルシューティング手順を提供します。

## よくある問題と解決方法

### 1. 定常状態計算の収束失敗

#### 症状
```
ValueError: SS comp failed: The iteration is not making good progress, max residual: X.XXXe-01
```

#### 原因と解決方法

**原因A: 税率変更幅が大きすぎる**
```python
# 問題のあるコード
reform = TaxReform('極端な改革', tau_c=0.25)  # 10% → 25%の大幅増税

# 解決方法: 段階的実装
reform = TaxReform('段階的改革', tau_c=0.15, implementation='phased', phase_in_periods=8)
```

**原因B: 複数税制の同時変更**
```python
# 問題のあるコード  
reform = TaxReform('同時改革', tau_c=0.15, tau_l=0.30, tau_k=0.40)

# 解決方法: 個別テスト後の組み合わせ
reform1 = TaxReform('消費税のみ', tau_c=0.15)
reform2 = TaxReform('労働税のみ', tau_l=0.30)
# 各々成功確認後に組み合わせ
```

**原因C: 「死の谷」範囲（1.5-2.5%ポイント）**
- 自動的に税制調整初期推定値が適用されるため、通常は問題なし
- 依然として問題がある場合は、初期推定値を手動指定：

```python
# 手動初期推定値の指定
baseline_ss = model.compute_steady_state()
reform_ss = reform_model.compute_steady_state(baseline_ss=baseline_ss)
```

### 2. 非現実的な計算結果

#### 症状
- 消費がGDPの数倍になる
- 負の税収
- 極端な変数値の変化

#### 診断方法
```python
# 結果の経済的妥当性チェック
def check_economic_realism(baseline_ss, reform_ss):
    print("経済的妥当性チェック:")
    
    # 主要比率の確認
    c_y_baseline = baseline_ss.C / baseline_ss.Y
    c_y_reform = reform_ss.C / reform_ss.Y
    print(f"C/Y比率: {c_y_baseline:.3f} → {c_y_reform:.3f}")
    
    if c_y_reform > 1.0:
        print("⚠️ 消費/GDP比率が100%超過 - 非現実的")
    
    # 税収の確認
    if reform_ss.T_total_revenue < 0:
        print("⚠️ 総税収が負の値 - モデル設定に問題")
    
    # 変化率の確認
    for var in ['Y', 'C', 'L']:
        change = (getattr(reform_ss, var) / getattr(baseline_ss, var) - 1) * 100
        print(f"{var}変化率: {change:+.1f}%")
        if abs(change) > 50:
            print(f"⚠️ {var}の変化が極端（{change:+.1f}%）")
```

#### 解決方法
1. **パラメータ校正の確認**
   ```python
   # config/parameters.json の主要パラメータチェック
   params = ModelParameters.from_json('config/parameters.json')
   print(f"減価償却率（四半期）: {params.delta}")  # 0.01程度が適切
   print(f"K/Y比率（年率）: {params.ky_ratio}")   # 8.0程度が適切
   ```

2. **初期推定値の改善**
   ```python
   # より保守的な初期推定値を手動作成
   initial_guess = {}
   for var in model.endogenous_vars_solve:
       baseline_val = getattr(baseline_ss, var)
       # 小さな摂動を加える
       initial_guess[var] = baseline_val * (1 + 0.01 * np.random.normal())
   ```

### 3. Blanchard-Kahn条件の違反

#### 症状
```
Warning: Blanchard-Kahn conditions not satisfied.
This may indicate model indeterminacy or non-existence of solution.
```

#### 対処方法
これは線形化段階の問題でありIssue #6（定常状態計算）とは異なります：

1. **定常状態の確認**
   ```python
   # まず定常状態が正常に計算されているか確認
   try:
       reform_ss = reform_model.compute_steady_state(baseline_ss=baseline_ss)
       print("✅ 定常状態計算は成功")
   except Exception as e:
       print(f"❌ 定常状態計算が失敗: {e}")
       return
   ```

2. **線形化問題への対処**
   - Blanchard-Kahn条件の違反は通常、モデルの構造的問題を示す
   - 税制改革の規模を小さくして再試行
   - 必要に応じて線形化専門の診断を実施

### 4. メモリ・パフォーマンス問題

#### 症状
- 計算時間が異常に長い
- メモリ使用量の増大
- プロセスの停止

#### 解決方法
```python
# 軽量化設定
reform_light = TaxReform(
    '軽量テスト',
    tau_c=0.11,
    implementation='permanent'  # 'phased'より高速
)

# 短期間シミュレーション
result = simulator.simulate_reform(reform_light, periods=20)  # 100→20に短縮
```

## 診断スクリプトの使用方法

### 包括的診断
```bash
# 全体的な問題診断
uv run python tests/debug_issue6_steady_state.py

# 特定の税制改革の詳細分析
uv run python tests/debug_2pp_failure.py
```

### 段階別診断
```python
# Step 1: ベースライン確認
python -c "
from src.dsge_model import DSGEModel, ModelParameters
params = ModelParameters.from_json('config/parameters.json')
model = DSGEModel(params)
ss = model.compute_steady_state()
print('ベースライン成功')
"

# Step 2: 簡単な税制改革テスト
python -c "
from src.dsge_model import DSGEModel, ModelParameters
params = ModelParameters.from_json('config/parameters.json')
baseline_model = DSGEModel(params)
baseline_ss = baseline_model.compute_steady_state()
params.tau_c = 0.11
reform_model = DSGEModel(params)
reform_ss = reform_model.compute_steady_state(baseline_ss=baseline_ss)
print('1pp改革成功')
"
```

## 税制改革設計のベストプラクティス

### 1. 段階的アプローチ
```python
# 推奨: 小さな変更から開始
reforms = [
    TaxReform('段階1', tau_c=0.105),  # 0.5pp
    TaxReform('段階2', tau_c=0.11),   # 1.0pp  
    TaxReform('段階3', tau_c=0.12),   # 2.0pp
    TaxReform('最終目標', tau_c=0.15)  # 5.0pp
]

for reform in reforms:
    try:
        result = simulator.simulate_reform(reform)
        print(f"✅ {reform.name} 成功")
    except Exception as e:
        print(f"❌ {reform.name} 失敗: {e}")
        break  # 失敗時点で停止
```

### 2. 収益中立改革の活用
```python
# 一つの税率を上げ、他を下げる収益中立改革
revenue_neutral_reform = TaxReform(
    '収益中立改革',
    tau_c=0.12,   # 消費税+2pp
    tau_l=0.18    # 労働税-2pp
)
```

### 3. 実装方式の選択
```python
# 大きな変更は段階的実装を推奨
large_reform = TaxReform(
    '大規模改革',
    tau_c=0.15,
    implementation='phased',    # 段階的実施
    phase_in_periods=8         # 2年かけて実施
)

# 小さな変更は恒久的実装でOK
small_reform = TaxReform(
    '小規模改革', 
    tau_c=0.11,
    implementation='permanent'  # 即座に実施
)
```

## パフォーマンス最適化

### 1. 計算設定の調整
```python
# 高速設定（精度との trade-off）
simulator_fast = EnhancedTaxSimulator(model)
result = simulator_fast.simulate_reform(
    reform,
    periods=20,           # 短期間
    compute_welfare=False # 厚生計算をスキップ
)

# 高精度設定（時間がかかる）
result = simulator_fast.simulate_reform(
    reform,
    periods=100,          # 長期間
    compute_welfare=True  # 厚生計算を実施
)
```

### 2. 並列処理の活用
```python
# 複数改革の並列テスト
from concurrent.futures import ThreadPoolExecutor

def test_reform(reform):
    try:
        result = simulator.simulate_reform(reform, periods=20)
        return f"✅ {reform.name}"
    except Exception as e:
        return f"❌ {reform.name}: {str(e)[:30]}"

reforms = [
    TaxReform('改革A', tau_c=0.11),
    TaxReform('改革B', tau_c=0.12),
    TaxReform('改革C', tau_l=0.22),
]

with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(test_reform, reforms))
    
for result in results:
    print(result)
```

## エラーメッセージ対応表

| エラーメッセージ | 原因 | 解決方法 |
|-----------------|------|----------|
| `SS comp failed: max residual: X.XXXe-01` | 定常状態収束失敗 | 税率変更幅を小さくする、初期推定値を調整 |
| `tau_k >= 0.95` | 資本税率が高すぎる | 80%以下に設定 |
| `Blanchard-Kahn conditions not satisfied` | 線形化問題 | 定常状態確認後、改革規模を縮小 |
| `ImprovedLinearizedDSGE object has no attribute` | 線形化オブジェクト問題 | 別issue、定常状態は成功している |

## 最新の改善事項

### Issue #6解決により改善された点
1. **自動初期推定値選択**: 税制変更幅に基づく最適戦略の自動選択
2. **「死の谷」対策**: 1.5-2.5%ポイント範囲の特別処理
3. **ソルバー最適化**: LM法の優先使用による収束性向上
4. **パラメータ安定化**: 極値防止と境界設定

### 今後の機能拡張計画
1. **感度分析機能**: パラメータ変更の影響度分析
2. **シナリオ比較機能**: 複数改革案の自動比較
3. **リスク評価機能**: 不確実性下での改革効果分析

---

**更新履歴**
- 2025年6月: Issue #6解決を受けて初版作成
- 技術担当: Daisuke Yoda
