# 日本税制シミュレーター - 実用コード例集

**重要**: これらの例は実際にテストされた日本税制シミュレーターのAPIのみを使用しています。すべてのコードスニペットは動作確認済みです。

## クイックスタート例（テスト済み）

```python
from src.dsge_model import DSGEModel, ModelParameters
from src.tax_simulator import EnhancedTaxSimulator, TaxReform

# 1. ベースラインモデルの読み込み（必須：定常状態の設定）
params = ModelParameters.from_json('config/parameters.json')
model = DSGEModel(params)
steady_state = model.compute_steady_state()
model.steady_state = steady_state  # 重要：これを設定する必要があります

# 2. シミュレーター作成
simulator = EnhancedTaxSimulator(
    model, 
    use_simple_linearization=True,  # True=安定版、False=研究用
    research_mode=False             # False=警告少なめ
)

# 3. 税制改革の定義
reform = TaxReform(
    name="消費税+1%",
    tau_c=0.11,  # 10% → 11%
    implementation='permanent'
)

# 4. シミュレーション実行
results = simulator.simulate_reform(reform, periods=8)

# 5. 結果分析（実際の属性を使用）
print(f"シミュレーション: {results.name}")
print(f"厚生変化: {results.welfare_change:.2%}")

# GDP影響計算（手動計算）
baseline_gdp = results.baseline_path['Y'].mean()
reform_gdp = results.reform_path['Y'].mean()
gdp_impact = (reform_gdp / baseline_gdp - 1) * 100

print(f"GDP影響: {gdp_impact:.2f}%")

# パスで利用可能な変数
print(f"利用可能な変数: {list(results.baseline_path.columns)}")
# 出力: ['Y', 'C', 'I', 'L', 'K', 'G']
```

## 動作確認済み可視化例

```python
import matplotlib.pyplot as plt

# 実際に利用可能な列名を使用
variables = ['Y', 'C', 'I', 'L']  # 簡単モデルでは'pi'は利用不可
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for i, var in enumerate(variables):
    row, col = i // 2, i % 2
    ax = axes[row, col]
    
    # ベースラインと改革パスをプロット
    ax.plot(results.baseline_path[var], label='ベースライン', linestyle='--', alpha=0.7)
    ax.plot(results.reform_path[var], label='改革後', linewidth=2)
    ax.set_title(f'{var}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
plt.suptitle('税制改革の影響: 移行動学')
plt.tight_layout()
plt.show()
```

## 複数シナリオ比較（テスト済み）

```python
# シナリオ定義
scenarios = {
    'ベースライン': TaxReform(name="ベースライン", tau_c=0.10),
    '小幅増税': TaxReform(name="消費税+1%", tau_c=0.11),
    '大幅増税': TaxReform(name="消費税+2%", tau_c=0.12)
}

# 全シナリオ実行
results_dict = {}
for name, reform in scenarios.items():
    print(f"{name}を実行中...")
    results_dict[name] = simulator.simulate_reform(reform, periods=8)

# 比較表作成
import pandas as pd

comparison_data = []
for name, results in results_dict.items():
    baseline_gdp = results.baseline_path['Y'].mean()
    reform_gdp = results.reform_path['Y'].mean()
    gdp_change = (reform_gdp / baseline_gdp - 1) * 100
    
    baseline_cons = results.baseline_path['C'].mean()
    reform_cons = results.reform_path['C'].mean()
    cons_change = (reform_cons / baseline_cons - 1) * 100
    
    comparison_data.append({
        'シナリオ': name,
        'GDP変化_%': gdp_change,
        '消費変化_%': cons_change,
        '厚生変化_%': results.welfare_change * 100
    })

df = pd.DataFrame(comparison_data)
print("\n=== シナリオ比較 ===")
print(df.round(2))
```

## 研究グレード例（Klein線形化）

```python
# 研究用設定（重要：use_simple_linearization=False）
research_simulator = EnhancedTaxSimulator(
    model,
    use_simple_linearization=False,  # Klein手法を使用
    research_mode=True               # 研究用警告を有効化
)

# 最初に小さな改革をテスト
small_reform = TaxReform(
    name="研究テスト",
    tau_c=0.105,  # 小幅な0.5%変化
    implementation='permanent'
)

try:
    research_results = research_simulator.simulate_reform(small_reform, periods=8)
    print("✓ 研究用シミュレーション成功")
    print(f"厚生影響: {research_results.welfare_change:.3%}")
except Exception as e:
    print(f"研究用シミュレーション失敗: {e}")
    # Blanchard-Kahn条件が満たされない場合に発生する可能性があります
```

## 結果保存（実際のメソッド）

```python
# CSV出力（動作確認済み）
results.baseline_path.to_csv('baseline_simulation.csv')
results.reform_path.to_csv('reform_simulation.csv')

# 要約統計の保存
summary = {
    'reform_name': results.name,
    'welfare_change': results.welfare_change,
    'transition_periods': results.transition_periods,
    'gdp_impact': (results.reform_path['Y'].mean() / results.baseline_path['Y'].mean() - 1) * 100
}

import json
with open('simulation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
```

## パフォーマンステスト

```python
import time

# シミュレーション速度テスト
start_time = time.time()
results = simulator.simulate_reform(reform, periods=20)
duration = time.time() - start_time

print(f"シミュレーション時間: {duration:.2f} 秒")
print(f"メモリ使用量: ~{results.baseline_path.memory_usage().sum() / 1024**2:.1f} MB")
```

## エラー処理

```python
try:
    # パラメータに問題がある場合失敗する可能性があります
    results = simulator.simulate_reform(reform, periods=40)
    print("✓ シミュレーション成功")
except ValueError as e:
    if "SS comp failed" in str(e):
        print("定常状態計算失敗。税率変化を小さくしてみてください。")
    elif "Blanchard-Kahn" in str(e):
        print("モデル解が不安定。パラメータを確認してください。")
    else:
        print(f"シミュレーションエラー: {e}")
```

## 利用可能なメソッドと属性

実際のテストに基づき、`SimulationResults`オブジェクトは以下を提供します：

```python
# 確認済み属性
results.name                    # str: 改革名
results.baseline_path          # DataFrame: ベースライン時系列
results.reform_path           # DataFrame: 改革後時系列  
results.welfare_change        # float: 厚生影響
results.fiscal_impact         # Dict: 政府予算への影響
results.transition_periods    # int: 収束までの期間
results.steady_state_baseline # SteadyState: ベースライン均衡
results.steady_state_reform   # SteadyState: 改革後均衡

# 確認済みメソッド
results.summary_statistics()         # Dict: 統計要約
results.get_impulse_responses(vars)  # DataFrame: インパルス応答計算
results.get_peak_effects(vars)       # Dict: ピーク影響分析
results.compute_aggregate_effects()  # DataFrame: 集計統計
results.to_dict()                   # Dict: 完全な結果エクスポート
```

## 重要な注意事項

1. **必ず`model.steady_state`を設定**してください（計算後）
2. **簡単モードで利用可能な変数**: `['Y', 'C', 'I', 'L', 'K', 'G']`
3. **研究モード**はBlanchard-Kahn条件が満たされない場合失敗する可能性があります
4. **短期シミュレーション**（8-20期間）がより高速で安定です
5. **大幅な税率変更**（>2%）は収束問題を引き起こす可能性があります

## 使用禁止（存在しません）

❌ `results.plot_transition()`  
❌ `results.get_gdp_change()`  
❌ `results.export_excel()`  
❌ `results.plot_educational_summary()`  
❌ `simulator.sensitivity_analysis()`  
❌ `simulator.monte_carlo_simulation()`  

代わりに手動計算とmatplotlibを可視化に使用してください。