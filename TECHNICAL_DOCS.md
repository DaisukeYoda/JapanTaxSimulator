# 技術ドキュメント - 日本税制シミュレーター

## 目次

1. [モデルの数学的定式化](#1-モデルの数学的定式化)
    - 1.1 [家計部門 (Households) - Open Economy Extension](#11-家計部門-households---open-economy-extension)
    - 1.2 [企業部門 (Firms)](#12-企業部門-firms)
    - 1.3 [政府部門 (Government)](#13-政府部門-government)
    - 1.4 [中央銀行 (Central Bank)](#14-中央銀行-central-bank)
    - 1.5 [国際部門 (International Sector)](#15-国際部門-international-sector)
    - 1.6 [市場清算条件 (Market Clearing)](#16-市場清算条件-market-clearing)
2. [実装アーキテクチャ](#2-実装アーキテクチャ)
3. [パラメータ詳細](#3-パラメータ詳細)
4. [使用方法詳細](#4-使用方法詳細)
5. [計算手法](#5-計算手法)
6. [検証とテスト](#6-検証とテスト)
7. [トラブルシューティング](#7-トラブルシューティング)

---

## 1. モデルの数学的定式化

### 1.1 家計部門 (Households) - Open Economy Extension

#### 効用関数
代表的家計は以下の期待効用を最大化：
```
U = E₀ Σ(t=0 to ∞) βᵗ [log(Cₜ - hCₜ₋₁) - χNₜ^(1+1/σₗ)/(1+1/σₗ)]
```
ここで：
- `Cₜ`: 期間tの消費
- `h`: 習慣形成パラメータ (0 ≤ h < 1)
- `Nₜ`: 期間tの労働時間
- `χ`: 労働の不効用パラメータ
- `σₗ`: フリッシュ労働供給弾力性 (Frisch elasticity of labor supply)

#### 新たに導入される変数 (Open Economy)
- `b\*ₜ`: 期間`t`の終わりに家計が保有する実質純外国債券残高（外国消費財単位で表示）。これらの債券は期間`t`に購入され、期間`t+1`に収益を生む。
- `qₜ`: 実質為替レート。国内消費財で測った外国消費財の相対価格 (`qₜ = eₜ P\*ₜ / Pₜ`)。ここで `eₜ` は名目為替レート（自国通貨/外国通貨）、`P\*ₜ` は外国物価水準、`Pₜ` は国内物価水準。
- `R\*ₜ`: 期間`t`から`t+1`にかけて保有される外国債券の実質総利子率（外国財単位）。

#### 単純化のための仮定
- 外国債券の取得・売却に伴うポートフォリオ調整費用は存在しない。
- `λₜ`（実質資産の限界効用）、`qₜ`（実質為替レート）、`R\*ₜ`（外国実質利子率）間の関係によって暗黙的に捉えられるものを超える、明示的な国別リスクプレミアムは外国債券保有にはないと仮定する。
- 外国債券所得に対する税金はここでは明示的にモデル化しない。もしモデル化する場合、国内資本所得や労働所得への課税と同様の方法で外国債券のFOCに影響する。この演習では、外国利子所得は非課税であるか、`R\*ₜ`が税引き後の利率であると仮定する。

#### 実質予算制約 (Open Economy)
家計の期間`t`における実質予算制約は以下のように修正される：
```
(1+τc,t)Cₜ + Iₜ + bₜ + qₜ b\*ₜ = (1-τₗ,ₜ)WₜNₜ + (1-τₖ,ₜ)RkₜKₜ₋₁ + Rₜ₋₁bₜ₋₁ + qₜ R\*ₜ₋₁b\*ₜ₋₁ + Tₜ
```
ここで：
- 左辺（資金の使途）：
    - `(1+τc,t)Cₜ`: 実質消費支出（消費税 `τc,t` 込み）
    - `Iₜ`: 国内実物資本への実質投資
    - `bₜ`: 期間`t`に購入し、期間`t+1`に満期を迎える実質国内債券
    - `qₜ b\*ₜ`: 期間`t`に購入し（外国財単位）、期間`t+1`に満期を迎える実質外国債券の現在価値（国内財単位換算）
- 右辺（資金の源泉）：
    - `(1-τₗ,ₜ)WₜNₜ`: 実質税引後労働所得（労働所得税 `τₗ,ₜ`, 実質賃金 `Wₜ`, 労働 `Nₜ`）
    - `(1-τₖ,ₜ)RkₜKₜ₋₁`: 実質税引後資本レンタル所得（資本所得税 `τₖ,ₜ`, 実質レンタル率 `Rkₜ`, 期間`t`開始時の資本ストック `Kₜ₋₁`）
    - `Rₜ₋₁bₜ₋₁`: 期間`t-1`から`t`にかけて保有した国内債券`bₜ₋₁`からの実質総収益（総実質利子率 `Rₜ₋₁`）
    - `qₜ R\*ₜ₋₁b\*ₜ₋₁`: 期間`t-1`から`t`にかけて保有した外国債券`b\*ₜ₋₁`（外国財単位）からの実質総収益（`qₜ`を用いて国内財単位に換算、外国総実質利子率 `R\*ₜ₋₁`）
    - `Tₜ`: 政府からの一括移転
- `Wₜ`: 実質賃金. `Rkₜ`: 実質資本レンタル率.

#### 資本蓄積
資本蓄積方程式は変更なし：
```
Kₜ = (1-δ)Kₜ₋₁ + Iₜ[1 - ψ/2(Iₜ/Kₜ₋₁ - δ)²]
```
ここで：
- `δ`: 資本減耗率
- `ψ`: 投資調整費用パラメータ

#### 一次条件 (First-Order Conditions)

**消費のFOC (`Cₜ`)**:
消費（および習慣形成）を実質資産の限界効用 `λₜ` および消費税 `τc,t` に関連付けるFOCは以下の通り：
```
1/(Cₜ - hCₜ₋₁) - βh Eₜ[1/(Cₜ₊₁ - hCₜ)] = λₜ(1+τc,t)
```
ここで `λₜ` は家計の実質予算制約のラグランジュ乗数であり、実質資産の限界効用を表す。

**労働供給のFOC (`Nₜ`)**:
これは変更なしと確認される (`Wₜ`は実質賃金)：
```
χNₜ^(1/σₗ) = λₜ(1-τₗ,ₜ)Wₜ/(1+τc,t)
```

**国内債券のFOC (`bₜ`)**:
国内債券のオイラー方程式も変更なしと確認される（`Rₜ`は`t`から`t+1`への国内債券の総実質利子率）：
```
λₜ = β Eₜ[λₜ₊₁ Rₜ]
```

**外国債券のFOC (`b\*ₜ`)**:
`b\*ₜ`の最適選択に関する新たなFOCは以下のように導出される：
```
λₜ qₜ = β Eₜ[λₜ₊₁ qₜ₊₁ R\*ₜ]
```
この条件は、リスク回避（`λ`を通じて）と期待（`Eₜ`を通じて）を調整したカバーなし金利パリティ（UIP）条件として解釈できる。

**投資選択 (`Iₜ`)**:
投資のFOCの構造は同様に`λₜ`を用いる。既存のドキュメントにおける投資選択のFOCは以下の通りであった（表記を`Rk`に修正）：
```
λₜ = β Et[λₜ₊₁((1-τₖ,ₜ₊₁)Rkₜ₊₁ + (1-δ))]
```
(注記: この投資のFOCは、投資調整費用(`ψ`)が存在するモデルにおいては単純化された形式です。より完全な形式ではトービンのQが関与します。しかし、このタスクの範囲は開放経済への移行に伴う変更に焦点を当てているため、既存の形式を維持します。)

### 1.2 企業部門 (Firms)

#### 生産関数
```
Yₜ = AₜKₜ₋₁^α Nₜ^(1-α)
```

ここで：
- `Aₜ`: 全要素生産性 (TFP)
- `α`: 資本分配率

#### 技術ショック
```
log(Aₜ) = ρₐ log(Aₜ₋₁) + εₐ,ₜ
```

#### 最終財企業の最適化問題
CES集計技術を用いて中間財を最終財に変換：

```
Yₜ = [∫₀¹ Yₜ(i)^((ε-1)/ε) di]^(ε/(ε-1))
```

#### 中間財企業の価格設定
カルボ型価格硬直性の下で、期間tに価格改定を行う企業i の最適価格：

```
log(Pₜ*(i)) = log(Pₜ) + (1-βθₚ)Σ(k=0 to ∞)(βθₚ)ᵏ Et[mcₜ₊ₖ]
```

ここで：
- `θₚ`: カルボパラメータ（価格据え置き確率）
- `mcₜ`: 実質限界費用

### 1.3 政府部門

#### 政府予算制約
```
Gₜ + Bₜ₋₁/πₜ + Tₜ = τc,t Cₜ + τₗ,ₜWₜNₜ + τₖ,ₜRₜKₜ₋₁ + τf,t Πₜ + Bₜ
```

ここで：
- `Gₜ`: 政府支出
- `τf,t`: 法人税率
- `Πₜ`: 企業利潤

#### 財政ルール
債務安定化のための税率調整ルール：

```
τₗ,ₜ = τₗ + φᵦ(Bₜ₋₁/Yₜ₋₁ - b̄)
```

ここで：
- `τₗ`: 定常状態の労働所得税率
- `φᵦ`: 財政反応係数
- `b̄`: 目標債務GDP比

### 1.4 中央銀行

#### テイラー則
```
iₜ = ρᵣiₜ₋₁ + (1-ρᵣ)[φπ(πₜ - π̄) + φᵧ(Yₜ - Ȳₜ)] + εᵣ,ₜ
```

ここで：
- `iₜ`: 名目金利
- `ρᵣ`: 金利平滑化パラメータ
- `φπ, φᵧ`: テイラー則係数
- `π̄`: インフレ目標
- `Ȳₜ`: 潜在GDP

### 1.5 国際部門 (International Sector)

このセクションでは、開放経済モデルにおける国際的な取引に関連する主要な変数と方程式を定義します。

#### 主要変数とパラメータ（国際部門）

*   **主要変数:**
    *   `Y\*ₜ`: 実質外国産出 (外生的)。
    *   `Aₜ`: 国内吸収 (`Cₜ + Iₜ + Gₜ`)。
    *   `IMₜ`: 国内経済による財・サービスの実質輸入（国内財単位で評価）。
    *   `EXₜ`: 国内経済による財・サービスの実質輸出（国内財単位で評価）。
    *   `NXₜ`: 実質純輸出 (`EXₜ - IMₜ`)。
    *   （`qₜ`, `b\*ₜ`, `R\*ₜ` は家計部門の拡張で導入済み）

*   **主要パラメータ:**
    *   `α_m`: 国内吸収における輸入のシェアパラメータ。
    *   `η_im`: 輸入の価格弾力性 (`qₜ`に対する)。
    *   `α_x`: 輸出のスケールパラメータ。
    *   `ϕ_ex`: 輸出の外国所得弾力性 (`Y\*ₜ`に対する)。
    *   `η_ex`: 輸出の価格弾力性 (`qₜ`に対する)。
    *   `Y\*_ss`: 外国産出の定常状態レベル。
    *   `ρ_y\*`: 外国産出の対数 `log(Y\*ₜ)` のAR(1)係数。
    *   `σ_y\*`: `log(Y\*ₜ)` へのショック `ε_y\*,ₜ` の標準偏差。

#### 国際部門の方程式

**a. 国内吸収の恒等式 (`Aₜ`):**
国内吸収は、国内の消費、投資、政府支出の合計です。
```
Aₜ = Cₜ + Iₜ + Gₜ
```

**b. 輸入需要関数 (`IMₜ`):**
輸入は国内吸収 (`Aₜ`) と実質為替レート (`qₜ`) に依存します。`qₜ` の上昇（外国財の相対価格上昇）は輸入需要を減少させます。
```
IMₜ = α_m * Aₜ * qₜ^(-η_im)
```
- `α_m > 0`: 輸入シェアパラメータ。
- `η_im > 0`: 輸入の価格弾力性。

**c. 輸出需要関数 (`EXₜ`):**
輸出は外国産出 (`Y\*ₜ`) と実質為替レート (`qₜ`) に依存します。`qₜ` の上昇（国内財の対外国相対価格低下）は輸出需要を増加させます。
```
EXₜ = α_x * (Y\*ₜ)^ϕ_ex * qₜ^(η_ex)
```
- `α_x > 0`: 輸出スケールパラメータ。
- `ϕ_ex > 0`: 輸出の外国所得弾力性。
- `η_ex > 0`: 輸出の価格弾力性。

**d. 外国産出プロセス (`Y\*ₜ`):**
外国産出は対数線形の外生AR(1)プロセスに従うと仮定されます。
```
log(Y\*ₜ) = (1-ρ_y\*)log(Y\*_ss) + ρ_y\* log(Y\*ₜ₋₁) + ε_y\*,ₜ
```
ここで `ε_y\*,ₜ ~ N(0, σ_y\*²)` は外国産出へのショック、`Y\*_ss` は外国産出の定常状態レベル、`ρ_y\*` は持続性パラメータです。

**e. 純輸出の恒等式 (`NXₜ`):**
純輸出は、国内財単位で評価された輸出と輸入の差です。
```
NXₜ = EXₜ - IMₜ
```

**f. 実質為替レートの動学 (`qₜ`):**
実質為替レート `qₜ` は `eₜ P\*ₜ / Pₜ` と定義されます。その動学は、家計の最適化問題から導かれるカバーなし金利パリティ（UIP）条件によって決定されます（外国債券 `b\*ₜ` のFOC）。
```
λₜ qₜ = β Eₜ[λₜ₊₁ qₜ₊₁ R\*ₜ]
```
これは期待相対リターンとして次のように書けます。
```
1 = Eₜ[β * (λₜ₊₁/λₜ) * (qₜ₊₁/qₜ) * R\*ₜ]
```

**g. 純外国資産 (NFA) の蓄積 / 国際収支:**
純外国資産（ここでは外国財単位で表示される家計保有の外国債券 `b\*ₜ` で代表される）の蓄積は、経常収支によって決定されます。経常収支は純輸出と外国資産からの純所得から構成されます。NFA蓄積方程式は、国内財単位でのフローの一貫性を確保して次のように記述されます。
```
qₜ b\*ₜ = qₜ R\*ₜ₋₁ b\*ₜ₋₁ + NXₜ
```
この式は、期間 `t` 末の純外国資産の価値（国内財換算 `qₜ b\*ₜ`）が、前期から保有する純外国資産の価値とそのグロスリターン（当期の為替レート `qₜ` で換算 `qₜ R\*ₜ₋₁ b\*ₜ₋₁`）に純輸出 (`NXₜ`) を加えたものに等しいことを示します。これは国の対外予算制約を表します。
外国財単位でのNFAの変動を示すと次のようになります。
```
b\*ₜ = R\*ₜ₋₁ b\*ₜ₋₁ + (NXₜ / qₜ)
```

### 1.6 市場清算条件

#### 財市場 (Open Economy)
開放経済における財市場の清算条件は、国内総生産 `Yₜ` が消費 `Cₜ`、投資 `Iₜ`、政府支出 `Gₜ`、および純輸出 `NXₜ` の合計に等しいことを示します。
```
Yₜ = Cₜ + Iₜ + Gₜ + NXₜ
```

#### 労働市場
```
Nₜ = ∫₀¹ Nₜ(i) di
```

---

## 2. 実装アーキテクチャ

### 2.1 コアクラス構造

#### DSGEModel クラス
```python
class DSGEModel:
    """動学的確率的一般均衡モデルの基底クラス"""
    
    def __init__(self, parameters: ModelParameters)
    def compute_steady_state(self) -> SteadyState
    def get_model_equations(self) -> List[Callable]
    def linearize(self) -> LinearizedModel
```

#### TaxPolicySimulator クラス
```python
class TaxPolicySimulator:
    """税制政策シミュレーション機能"""
    
    def simulate_reform(self, reform: TaxReform, periods: int) -> SimulationResults
    def compare_scenarios(self, scenarios: List[TaxReform]) -> ComparisonResults
    def welfare_analysis(self, reform: TaxReform) -> WelfareResults
```

### 2.2 データフロー

```
パラメータ読み込み (JSON) 
    ↓
モデル初期化 (DSGEModel)
    ↓
定常状態計算 (SteadyState)
    ↓
線形化 (LinearizedDSGE)
    ↓
シミュレーション (TaxPolicySimulator)
    ↓
結果出力・可視化
```

### 2.3 モジュール構成

#### src/dsge_model.py
- **ModelParameters**: パラメータ管理
- **SteadyState**: 定常状態計算
- **DSGEModel**: メインモデルクラス

#### src/tax_simulator.py
- **TaxReform**: 税制改革の定義
- **TaxPolicySimulator**: シミュレーション実行
- **WelfareAnalyzer**: 厚生分析

#### src/linearization_improved.py
- **ImprovedLinearizedDSGE**: 高度な線形化手法
- **ImpulseResponseComputer**: インパルス応答計算

---

## 3. パラメータ詳細

### 3.1 家計パラメータ

| パラメータ | 記号 | 標準値 | 範囲 | 説明 |
|-----------|------|--------|------|------|
| 割引因子 | β | 0.99 | [0.95, 0.999] | 四半期ベース、年率約4% |
| 異時点間代替弾力性 | σc | 1.5 | [0.5, 3.0] | 消費の平滑化度合い |
| フリッシュ弾力性 | σₗ | 2.0 | [0.5, 5.0] | 労働供給の弾力性 |
| 習慣形成 | h | 0.6 | [0.0, 0.9] | 過去消費への依存度 |
| 労働不効用 | χ | 3.0 | [1.0, 10.0] | 労働の限界不効用 |

### 3.2 企業パラメータ

| パラメータ | 記号 | 標準値 | 範囲 | 説明 |
|-----------|------|--------|------|------|
| 資本分配率 | α | 0.33 | [0.25, 0.40] | 資本の生産寄与度 |
| 減耗率 | δ | 0.025 | [0.015, 0.035] | 四半期ベース、年率約10% |
| 価格硬直性 | θₚ | 0.75 | [0.5, 0.9] | 価格据え置き確率 |
| 代替弾力性 | ε | 6.0 | [3.0, 11.0] | 財間の代替可能性 |
| 調整費用 | ψ | 4.0 | [1.0, 10.0] | 投資の調整コスト |

### 3.3 政策パラメータ

#### 財政政策
| パラメータ | 記号 | 標準値 | 説明 |
|-----------|------|--------|------|
| 政府支出GDP比 | gy | 0.20 | 20%（日本の実績） |
| 債務GDP比 | by | 2.0 | 200%（日本の実績） |
| 財政反応係数 | φᵦ | 0.1 | 債務安定化の強度 |

#### 金融政策
| パラメータ | 記号 | 標準値 | 説明 |
|-----------|------|--------|------|
| インフレ反応 | φπ | 1.5 | テイラー原則を満たす |
| 産出反応 | φᵧ | 0.125 | 適度な実体経済配慮 |
| 金利平滑化 | ρᵣ | 0.8 | 高い持続性 |

### 3.4 税制パラメータ

#### ベースライン税率
| 税目 | 記号 | 標準値 | 実績参考 |
|------|------|--------|---------|
| 消費税 | τc | 0.10 | 10%（2019年〜） |
| 労働所得税 | τₗ | 0.20 | 約20%（平均実効税率） |
| 資本所得税 | τₖ | 0.25 | 約25%（配当・利子等） |
| 法人税 | τf | 0.30 | 約30%（実効税率） |

---

## 4. 使用方法詳細

### 4.1 基本的なワークフロー

#### ステップ1: 環境設定
```python
import sys
sys.path.append('src')

from dsge_model import DSGEModel, ModelParameters
from tax_simulator import TaxPolicySimulator, TaxReform
import json

# パラメータ読み込み
with open('config/parameters.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
```

#### ステップ2: モデル初期化
```python
# パラメータ設定
params = ModelParameters()
params.update_from_dict(config['model_parameters'])

# モデル作成
model = DSGEModel(params)

# 定常状態計算
steady_state = model.compute_steady_state()
print(f"定常状態GDP: {steady_state.Y:.4f}")
```

#### ステップ3: シミュレーション設定
```python
# シミュレーター初期化
simulator = TaxPolicySimulator(model)

# 税制改革の定義
reform_scenarios = [
    TaxReform(
        name="消費税15%",
        tau_c=0.15,
        implementation='permanent'
    ),
    TaxReform(
        name="所得税減税",
        tau_l=0.15,
        implementation='permanent'
    )
]
```

#### ステップ4: シミュレーション実行
```python
# 単一シナリオ
results = simulator.simulate_reform(reform_scenarios[0], periods=40)

# 複数シナリオ比較
comparison = simulator.compare_scenarios(reform_scenarios)

# 結果の可視化
simulator.plot_results(results, variables=['Y', 'C', 'I', 'pi'])
```

### 4.2 高度な分析機能

#### 厚生分析
```python
welfare_results = simulator.welfare_analysis(reform_scenarios[0])
print(f"消費等価変分: {welfare_results.consumption_equivalent_variation:.2%}")
```

#### 感応度分析
```python
# パラメータの感応度テスト
sensitivity_params = ['sigma_c', 'theta_p', 'phi_pi']
sensitivity_results = simulator.sensitivity_analysis(
    reform_scenarios[0], 
    sensitivity_params,
    variation_range=0.2  # ±20%
)
```

#### モンテカルロシミュレーション
```python
# 不確実性を考慮した分析
mc_results = simulator.monte_carlo_simulation(
    reform_scenarios[0],
    n_simulations=1000,
    shock_types=['technology', 'government_spending']
)
```

### 4.3 カスタム税制改革の定義

#### 段階的導入
```python
phased_reform = TaxReform(
    name="段階的消費税増税",
    tau_c=0.15,
    implementation='phased',
    phase_in_periods=12  # 3年間で段階的実施
)
```

#### 一時的変更
```python
temporary_reform = TaxReform(
    name="一時的所得税減税",
    tau_l=0.15,
    implementation='temporary',
    duration=8  # 2年間のみ
)
```

#### 複合的改革
```python
comprehensive_reform = TaxReform(
    name="包括的税制改革",
    tau_c=0.12,    # 消費税12%
    tau_l=0.18,    # 所得税18%
    tau_k=0.20,    # 資本所得税20%
    tau_f=0.25,    # 法人税25%
    implementation='phased',
    phase_in_periods=16  # 4年間で実施
)
```

---

## 5. 計算手法

### 5.1 定常状態計算

#### 非線形連立方程式の解法
```python
def solve_steady_state(params):
    """非線形ソルバーによる定常状態計算"""
    
    def steady_state_equations(vars):
        # 変数の展開
        Y, C, I, K, N, W, R, pi, lambda_h = vars
        
        # 方程式システム
        equations = [
            # 家計の最適化条件
            chi * N**(1/params.sigma_l) - lambda_h * (1-params.tau_l) * W / (1+params.tau_c),
            # 企業の最適化条件
            W - (1-params.alpha) * Y / N,
            R - params.alpha * Y / K,
            # 資源制約
            Y - C - I - params.gy_ratio * Y,
            # 資本蓄積（定常状態）
            I - params.delta * K,
            # 生産関数
            Y - K**params.alpha * N**(1-params.alpha),
            # フィッシャー方程式
            1 + params.r_steady - (1 + params.i_steady) / pi,
            # オイラー方程式
            lambda_h - params.beta * lambda_h * (1 + params.r_steady),
            # 正規化条件
            pi - 1.0
        ]
        
        return equations
    
    # 初期推定値
    initial_guess = [1.0, 0.6, 0.2, 10.0, 0.33, 1.0, 0.04, 1.0, 1.0]
    
    # 数値解法
    result = optimize.fsolve(steady_state_equations, initial_guess)
    
    return result
```

### 5.2 線形化手法

#### 対数線形化による近似
モデル方程式 `f(x_{t+1}, x_t, z_t) = 0` を定常状態 `(x̄, z̄)` の周りで線形化：

```
f_x₊₁ · (x_{t+1} - x̄) + f_x · (x_t - x̄) + f_z · (z_t - z̄) = 0
```

#### 状態空間表現への変換
```python
def linearize_model(model, steady_state):
    """モデルの線形化と状態空間表現への変換"""
    
    # ヤコビアン行列の数値計算
    jacobian = compute_jacobian(model.equations, steady_state)
    
    # 状態空間形式: A·E[x_{t+1}] = B·x_t + C·z_t
    A = jacobian['forward']
    B = jacobian['current']  
    C = jacobian['shocks']
    
    return A, B, C
```

### 5.3 動学解法

#### Klein (2000) 手法による解法
```python
def solve_linear_system(A, B, C):
    """Klein手法による線形有理期待モデルの解法"""
    
    # 一般化固有値問題の解法
    eigenvals, eigenvecs = linalg.eig(B, A)
    
    # 安定・不安定固有値の分離
    stable = np.abs(eigenvals) < 1.0
    unstable = ~stable
    
    # ブランチャード・カーン条件の確認
    n_unstable = np.sum(unstable)
    n_forward = count_forward_variables(model)
    
    if n_unstable != n_forward:
        raise ValueError("ブランチャード・カーン条件が満たされません")
    
    # 政策関数の計算
    # x_t = P·x_{t-1} + Q·z_t
    P, Q = compute_policy_functions(eigenvals, eigenvecs, A, B, C)
    
    return P, Q
```

### 5.4 インパルス応答計算

```python
def compute_impulse_response(P, Q, shock_index, periods=40):
    """インパルス応答関数の計算"""
    
    n_vars = P.shape[0]
    n_shocks = Q.shape[1]
    
    # 応答パスの初期化
    responses = np.zeros((periods, n_vars))
    
    # 初期ショック
    shock = np.zeros(n_shocks)
    shock[shock_index] = 1.0  # 1標準偏差ショック
    
    # 動学的応答の計算
    state = Q @ shock  # 初期応答
    
    for t in range(periods):
        responses[t, :] = state
        state = P @ state  # 次期状態
    
    return responses
```

---

## 6. 検証とテスト

### 6.1 数値的妥当性の確認

#### ユニットテスト例
```python
def test_steady_state_calculation():
    """定常状態計算のテスト"""
    params = ModelParameters()
    model = DSGEModel(params)
    ss = model.compute_steady_state()
    
    # 基本的な整合性チェック
    assert ss.Y > 0, "GDPは正の値である必要があります"
    assert 0 < ss.C/ss.Y < 1, "消費GDP比は0と1の間である必要があります"
    assert 0 < ss.I/ss.Y < 1, "投資GDP比は0と1の間である必要があります"
    
    # 定常状態方程式の確認
    residuals = model.evaluate_equations(ss)
    assert np.allclose(residuals, 0, atol=1e-8), "定常状態方程式の残差が大きすぎます"

def test_linearization_accuracy():
    """線形化精度のテスト"""
    # 小さな摂動に対する線形近似の精度確認
    pass

def test_impulse_responses():
    """インパルス応答の妥当性テスト"""
    # 既知の理論的性質の確認
    pass
```

### 6.2 経済的妥当性の確認

#### マクロ経済学的整合性
- GDP構成要素の合計がGDPと一致
- 労働分配率が現実的な範囲内
- 金利の均衡条件の成立

#### 税制変更の妥当な反応
- 消費税増税 → 消費減少、物価上昇
- 所得税減税 → 労働供給増加、消費増加
- 法人税減税 → 投資増加

### 6.3 感応度分析

```python
def sensitivity_test(base_params, param_name, variation_range=0.2):
    """パラメータ感応度テスト"""
    
    results = {}
    
    for variation in [-variation_range, 0, variation_range]:
        # パラメータ変更
        test_params = base_params.copy()
        original_value = getattr(test_params, param_name)
        setattr(test_params, param_name, original_value * (1 + variation))
        
        # モデル再計算
        model = DSGEModel(test_params)
        ss = model.compute_steady_state()
        
        results[variation] = ss
    
    return results
```

---

## 7. トラブルシューティング

### 7.1 よくある問題と解決法

#### 定常状態が解けない
**症状**: `scipy.optimize.fsolve` が収束しない

**原因と対策**:
1. **初期値の問題**: より良い初期推定値を設定
2. **パラメータの問題**: 極端なパラメータ値を避ける
3. **方程式の誤り**: モデル方程式の再確認

```python
# 改善例：複数の初期値を試行
initial_guesses = [
    [1.0, 0.6, 0.2, 10.0, 0.33, 1.0, 0.04, 1.0, 1.0],
    [0.8, 0.5, 0.25, 12.0, 0.30, 0.9, 0.05, 1.0, 1.1],
    [1.2, 0.7, 0.15, 8.0, 0.35, 1.1, 0.03, 1.0, 0.9]
]

for guess in initial_guesses:
    try:
        result = optimize.fsolve(equations, guess)
        if check_solution_validity(result):
            break
    except:
        continue
```

#### ブランチャード・カーン条件違反
**症状**: 一意解が存在しない、または解が不安定

**対策**:
1. **パラメータ調整**: テイラー原則の確認（φπ > 1）
2. **モデル仕様の見直し**: 前向き変数の数と不安定根の数の確認

#### 数値的不安定性
**症状**: 計算結果が発散、または NaN が発生

**対策**:
1. **スケーリング**: 変数の正規化
2. **許容誤差の調整**: 収束基準の緩和
3. **代替解法**: 異なる数値解法の試行

### 7.2 パフォーマンス最適化

#### 計算速度の改善
```python
# NumPy配列操作の最適化
@numba.jit  # JITコンパイル
def compute_jacobian_fast(equations, point):
    # 高速化されたヤコビアン計算
    pass

# 並列計算の活用
from multiprocessing import Pool

def parallel_monte_carlo(n_simulations):
    with Pool() as pool:
        results = pool.map(single_simulation, range(n_simulations))
    return results
```

#### メモリ使用量の最適化
```python
# 大きな行列の効率的な処理
from scipy.sparse import csr_matrix

def sparse_matrix_operations(A, B):
    # 疎行列を用いた計算
    A_sparse = csr_matrix(A)
    result = A_sparse @ B
    return result.toarray()
```

### 7.3 デバッグ手法

#### 段階的検証
```python
def debug_model_step_by_step():
    """モデルの段階的デバッグ"""
    
    # ステップ1: パラメータ確認
    params = ModelParameters()
    print(f"パラメータ確認: β={params.beta}, σc={params.sigma_c}")
    
    # ステップ2: 定常状態計算
    model = DSGEModel(params)
    try:
        ss = model.compute_steady_state()
        print(f"定常状態計算成功: Y={ss.Y:.4f}")
    except Exception as e:
        print(f"定常状態計算エラー: {e}")
        return
    
    # ステップ3: 線形化
    try:
        linearized = model.linearize()
        print("線形化成功")
    except Exception as e:
        print(f"線形化エラー: {e}")
        return
    
    # ステップ4: 動学解法
    try:
        P, Q = linearized.solve()
        print("動学解法成功")
    except Exception as e:
        print(f"動学解法エラー: {e}")
        return
```

#### ログ出力の活用
```python
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def compute_steady_state_with_logging(params):
    logger.info("定常状態計算開始")
    try:
        result = solve_steady_state(params)
        logger.info(f"計算成功: GDP={result[0]:.4f}")
        return result
    except Exception as e:
        logger.error(f"計算失敗: {e}")
        raise
```

---

## 付録

### A. 主要数式一覧

[詳細な数式リストは別途LaTeX文書として作成可能]

### B. パラメータ校正詳細

[校正手法と日本経済データとの対応表]

### C. 参考文献

- **理論**: Galí (2015), Woodford (2003)
- **数値解法**: Klein (2000), Uhlig (1999)
- **日本経済**: 内閣府ESRI, 日本銀行調査統計局

### D. API リファレンス

[クラスとメソッドの詳細仕様]

---

**最終更新**: [日付]
**バージョン**: 1.0
**作成者**: Daisuke Yoda