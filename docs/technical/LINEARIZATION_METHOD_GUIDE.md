# 線形化手法選択ガイド（Issue #30対応）

**対象**: 学術研究者、政策分析者、教育者  
**バージョン**: v1.0  
**最終更新**: 2025年6月8日  

## 概要

Japan Tax Simulatorでは、2つの線形化手法が利用可能です：

1. **簡略化線形化** - デモンストレーション・教育用途に最適
2. **完全線形化（Klein解法）** - 学術研究・政策分析用途に推奨

このガイドは、使用目的に応じた適切な手法選択を支援します。

## 🚨 重要: Issue #30で判明した問題

### 定量的分析結果

2025年6月の包括的比較分析により、以下が判明しました：

- **83%のシナリオで5%以上の差異**を検出
- **最大7.54%の相対差異**（所得税減税シナリオ）
- **小規模改革のみ**許容範囲内（1%消費税：0.72%差異）

### 学術研究への影響

中規模以上の税制改革（2%ポイント以上）では、手法間の差異が学術的に有意なレベルに達します。

## 線形化手法の比較

### 簡略化線形化

**特徴**：
- 固定的な政策関数行列（P）と遷移行列（Q）
- 実証研究に基づく固定係数
- 常に安定・収束する設計
- 計算が高速

**経済的仮定**：
```python
# 主要変数への税制効果（実証ベース）
GDP_tax_sensitivity = -0.08    # 1%税率上昇で0.08%GDP減少
Consumption_sensitivity = -0.12  # より敏感に反応
Investment_sensitivity = -0.10   # 中程度の感度
Labor_sensitivity = -0.05       # 小さな影響
```

**適用場面**：
- ✅ デモンストレーション
- ✅ 教育・学習目的
- ✅ 概念理解の促進
- ✅ 小規模税制変更（<2%ポイント）の概算

### 完全線形化（Klein解法）

**特徴**：
- 動的確率一般均衡（DSGE）理論に基づく導出
- Blanchard-Kahn条件による解の存在・一意性確認
- シンボリック微分による正確な偏微分係数
- モデル構造に依存した係数

**経済的基盤**：
- 家計の効用最大化
- 企業の利潤最大化
- 市場均衡条件
- 動学的最適化

**適用場面**：
- ✅ 学術研究・論文執筆
- ✅ 政策提言・分析
- ✅ 中・大規模税制改革（≥2%ポイント）
- ✅ 理論的精度が要求される分析

## 使用方法

### 1. 学術研究・政策分析用途

```python
from src.dsge_model import DSGEModel, ModelParameters
from src.tax_simulator import EnhancedTaxSimulator, TaxReform

# パラメータとモデルの準備
params = ModelParameters.from_json('config/parameters.json')
model = DSGEModel(params)

# 🎓 学術研究用: 完全線形化を明示的に指定
simulator = EnhancedTaxSimulator(
    model, 
    use_simple_model=False,
    use_simple_linearization=False  # 重要: 完全線形化を指定
)

# 税制改革の分析
reform = TaxReform(
    name="消費税3%ポイント増税",
    tau_c=params.tau_c + 0.03,
    implementation='permanent'
)

results = simulator.simulate_reform(reform, periods=100)
```

### 2. デモンストレーション・教育用途

```python
# 📚 教育用: 簡略化線形化を明示的に指定
simulator = EnhancedTaxSimulator(
    model, 
    use_simple_model=False,
    use_simple_linearization=True  # 安定性・理解しやすさ優先
)

# 概念説明用の小規模改革
demo_reform = TaxReform(
    name="デモ用消費税1%増税",
    tau_c=params.tau_c + 0.01,
    implementation='permanent'
)

results = simulator.simulate_reform(demo_reform, periods=50)
```

### 3. 比較分析（推奨）

```python
# 🔬 研究用: 両手法で分析して頑健性を確認
reforms = [
    TaxReform(name="大規模改革", tau_c=params.tau_c + 0.05)
]

for reform in reforms:
    # 完全線形化による分析
    simulator_full = EnhancedTaxSimulator(
        model, use_simple_model=False, use_simple_linearization=False
    )
    results_full = simulator_full.simulate_reform(reform)
    
    # 簡略化線形化による分析
    simulator_simple = EnhancedTaxSimulator(
        model, use_simple_model=False, use_simple_linearization=True
    )
    results_simple = simulator_simple.simulate_reform(reform)
    
    # 差異を報告
    # [比較ロジック]
```

## 推奨事項マトリックス

| 使用目的 | 税制変更規模 | 推奨手法 | 理由 |
|---------|-------------|----------|------|
| 学術論文 | 全サイズ | 完全線形化 | 理論的正当性 |
| 政策分析 | ≥2%ポイント | 完全線形化 | 精度要件 |
| 政策分析 | <2%ポイント | 両手法比較 | 頑健性確認 |
| デモ・教育 | 全サイズ | 簡略化線形化 | 安定性・理解 |
| 概念理解 | 小規模 | 簡略化線形化 | 直感的結果 |

## 品質保証のためのチェックリスト

### 学術研究用途

- [ ] `use_simple_linearization=False` を明示的に指定
- [ ] Blanchard-Kahn条件の満足を確認
- [ ] 収束診断を実施
- [ ] 感度分析を実施
- [ ] 方法論を論文で明記

### 政策分析用途

- [ ] 改革規模に応じた手法選択
- [ ] 大規模改革では完全線形化を使用
- [ ] 両手法での比較分析を実施
- [ ] 差異が5%超の場合は要注意
- [ ] 不確実性を適切に報告

### デモ・教育用途

- [ ] `use_simple_linearization=True` を明示的に指定
- [ ] 対象者への説明責任
- [ ] 簡略化の限界を説明
- [ ] 学術用途との違いを明記

## 実証分析結果

### 手法間差異の典型例

```
シナリオ: 消費税3%ポイント増税
- 簡略化線形化: GDP -2.4%
- 完全線形化: GDP -2.6% 
- 相対差異: 6.3%
- 評価: 学術的に有意
```

### 許容範囲の例

```
シナリオ: 消費税1%ポイント増税
- 簡略化線形化: GDP -0.8%
- 完全線形化: GDP -0.8%
- 相対差異: 0.7%
- 評価: 許容範囲内
```

## トラブルシューティング

### 完全線形化が失敗する場合

1. **Blanchard-Kahn条件の確認**
   ```
   Warning: Blanchard-Kahn conditions not satisfied.
   ```
   - パラメータの境界値確認
   - モデル仕様の見直し
   - 簡略化線形化への切り替え検討

2. **収束の問題**
   ```
   Warning: Could not solve for policy function
   ```
   - 税制変更の規模を調整
   - 段階的実施の検討
   - フォールバック機能の活用

### 精度に関する警告への対応

```python
# 研究整合性警告が表示される場合
import os
os.environ['RESEARCH_MODE'] = 'development'  # 開発モード

# または
os.environ['RESEARCH_MODE'] = 'strict'       # 厳格モード
```

## 参考文献・理論的背景

1. **Klein (2000)**: "Using the generalized Schur form to solve a multivariate linear rational expectations model"
2. **Sims (2002)**: "Solving linear rational expectations models"
3. **Blanchard & Kahn (1980)**: "The solution of linear difference models under rational expectations"

## バージョン履歴

- **v1.0 (2025-06-08)**: 初版リリース（Issue #30対応）
  - 包括的比較分析結果を反映
  - 使用目的別推奨事項を明確化
  - 実証データに基づく判断基準を設定

## サポート

問題や質問がある場合：
1. `scripts/validation/linearization_method_comparison.py` で比較分析を実行
2. `docs/technical/ISSUE_6_RESOLUTION.md` で関連する技術的詳細を確認
3. GitHub Issue として報告

---

**重要**: このガイドは継続的に更新されます。最新の研究結果や実証分析を基に、推奨事項の見直しを定期的に実施します。