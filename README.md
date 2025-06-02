# 日本税制シミュレーター (Japan Tax Simulator)

日本経済の税制変更がマクロ経済に与える影響を分析するための動学的確率的一般均衡（DSGE）モデルの実装

## 📖 概要

このプロジェクトは、日本経済の構造的特徴を反映した高精度なDSGEモデルを用いて、以下の税制の変更が経済に与える影響を定量的に分析するツールです：

- **消費税** (τc): 消費支出に対する課税
- **所得税** (τl): 労働所得に対する課税  
- **資本所得税** (τk): 資本所得に対する課税
- **法人税** (τf): 企業利潤に対する課税

## 🏗️ モデル構造

### ニューケインジアンDSGEモデル
本モデルは、以下の経済主体の最適化行動に基づく標準的なニューケインジアンフレームワークを採用：

#### 🏠 家計部門
- **効用関数**: 消費と労働からの効用（習慣形成を含む）
- **予算制約**: 各種税制を考慮した所得・支出・資産
- **最適化**: 生涯効用最大化

#### 🏭 企業部門
- **生産関数**: コブ＝ダグラス型（資本・労働入力）
- **価格設定**: カルボ型価格硬直性
- **投資決定**: トービンのqと資本調整費用

#### 🏛️ 政府部門
- **財政政策**: 政府支出、税収、債務管理
- **財政ルール**: 債務残高GDP比に応じた税率調整

#### 🏦 中央銀行
- **金融政策**: テイラー則による政策金利決定

### 📊 主要パラメータ（日本経済キャリブレーション）

| パラメータ | 値 | 説明 |
|-----------|---|------|
| β | 0.99 | 割引因子（四半期） |
| σc | 1.5 | 異時点間代替の弾力性 |
| σl | 2.0 | フリッシュ労働供給弾力性 |
| α | 0.33 | 資本分配率 |
| θp | 0.75 | カルボ価格硬直性 |
| φπ | 1.5 | テイラー則インフレ係数 |
| τc | 0.10 | 消費税率（ベースライン10%） |
| τl | 0.20 | 所得税率（平均） |

## 🚀 主要機能

### 1. 定常状態計算
- 全変数の長期均衡値を数値的に解法
- パラメータ変更時の新しい定常状態を自動計算

### 2. 動学分析
- モデルの対数線形化
- インパルス応答関数の計算・可視化
- 各種ショック（技術、財政、金融、税制）の影響分析

### 3. 税制変更シミュレーション
- **シナリオ設定**: 恒久的・一時的・段階的な税制変更
- **比較分析**: ベースラインとの差分を定量化
- **多角的評価**: GDP、消費、投資、税収、政府債務への影響

### 4. 高度な分析機能
- 厚生分析（消費等価変分）
- 政策の最適化
- 不確実性下での頑健性分析
- モンテカルロシミュレーション

## 📁 プロジェクト構造

```
JapanTaxSimulator/
├── README.md                          # このファイル
├── requirements.txt                   # 依存パッケージ
├── config/
│   └── parameters.json               # モデルパラメータ設定
├── src/
│   ├── __init__.py
│   ├── dsge_model.py                # 基本DSGEモデル実装
│   ├── linearization.py             # 線形化モジュール
│   ├── linearization_improved.py    # 改良版線形化
│   └── tax_simulator.py             # 税制シミュレーター
├── notebooks/
│   ├── tax_simulation_demo.ipynb    # 基本デモ
│   └── advanced_tax_simulation_demo.ipynb  # 高度な分析デモ
├── data/                             # データファイル
├── results/                          # 出力結果
├── test_model.py                     # モデルテスト
└── quick_check.py                    # クイックチェック
```

## 💻 インストールと実行

### 環境構築

```bash
# リポジトリをクローン
git clone [repository-url]
cd JapanTaxSimulator

# uv を使用（推奨・高速）
uv sync

# インストール確認
uv run python quick_check.py
```

※ uvがインストールされていない場合：
```bash
# macOS
brew install uv

# その他のOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 基本的な使用例

#### Python スクリプトでの実行

```python
from src.dsge_model import DSGEModel, ModelParameters
from src.tax_simulator import TaxPolicySimulator, TaxReform

# モデルの初期化
params = ModelParameters()
model = DSGEModel(params)

# シミュレーターの作成
simulator = TaxPolicySimulator(model)

# 消費税5%ポイント引き上げシナリオ
reform = TaxReform(
    name="消費税増税5%",
    tau_c=0.15,  # 10% → 15%
    implementation='permanent'
)

# シミュレーション実行
results = simulator.simulate_reform(reform, periods=40)

# 結果の可視化
simulator.plot_results(results)
```

#### Jupyter Notebookでの実行

```bash
# Jupyter Notebookを起動
uv run jupyter notebook

# ブラウザで以下のファイルを開く：
# - notebooks/tax_simulation_demo.ipynb（基本デモ）
# - notebooks/advanced_tax_simulation_demo.ipynb（高度な分析）
```

### クイックテスト

```bash
# モデルの基本動作確認
uv run python quick_check.py

# 詳細なテスト実行
uv run pytest
```

## 📈 主要な分析結果（例）

### 消費税5%ポイント引き上げの影響

| 指標 | 短期（1年後） | 中期（5年後） | 長期（定常状態） |
|------|---------------|---------------|------------------|
| GDP | -1.2% | -0.8% | -0.5% |
| 消費 | -2.1% | -1.4% | -1.0% |
| 投資 | -0.8% | -0.4% | -0.2% |
| 物価水準 | +4.5% | +4.8% | +5.0% |
| 総税収 | +2.8% | +3.2% | +3.5% |

※ 数値は例示であり、実際の結果はパラメータ設定により変動します

## 🔧 カスタマイゼーション

### パラメータ設定の変更

`config/parameters.json`ファイルを編集することで、モデルパラメータを調整できます：

```json
{
    "model_parameters": {
        "household": {
            "beta": 0.99,
            "sigma_c": 1.5,
            "sigma_l": 2.0
        },
        "firm": {
            "alpha": 0.33,
            "delta": 0.025
        }
    },
    "tax_parameters": {
        "baseline": {
            "tau_c": 0.10,
            "tau_l": 0.20
        }
    }
}
```

### 独自の税制シナリオ作成

```python
# カスタム税制改革の定義
custom_reform = TaxReform(
    name="包括的税制改革",
    tau_c=0.12,    # 消費税12%
    tau_l=0.18,    # 所得税18%
    tau_f=0.25,    # 法人税25%
    implementation='phased',  # 段階的実施
    phase_in_periods=8        # 2年間で段階的導入
)
```

## 📚 理論的背景

### 数学的定式化

#### 家計の最適化問題
```
max E₀ Σ(t=0 to ∞) βᵗ [log(Cₜ - hCₜ₋₁) - χNₜ^(1+1/σₗ)/(1+1/σₗ)]

subject to:
(1+τc)Cₜ + Iₜ + Bₜ ≤ (1-τₗ)WₜNₜ + (1-τₖ)RₜKₜ + Bₜ₋₁/πₜ + Tₜ
```

#### 企業の利潤最大化
```
max E₀ Σ(t=0 to ∞) Λₜ[(1-τf)(PₜYₜ - WₜNₜ - RₜKₜ) - ψ/2(Iₜ/Kₜ₋₁ - δ)²Kₜ₋₁]
```

### キャリブレーション手法

パラメータは以下の日本経済データとターゲットに基づきキャリブレーション：

1. **実質データ**: 内閣府GDP統計、日本銀行短観
2. **労働市場**: 総務省労働力調査、厚生労働省賃金統計
3. **財政データ**: 財務省財政統計、税収実績
4. **金融データ**: 日本銀行政策金利、長期金利

## 🧪 検証とテスト

### モデル妥当性の確認

1. **ブランチャード・カーン条件**: 一意安定解の存在確認
2. **定常状態値**: 日本経済の実績値との適合性
3. **インパルス応答**: 既存研究との比較
4. **感応度分析**: パラメータに対する頑健性

### 実証分析との比較

- 消費税増税時（1989年、1997年、2014年、2019年）の実績データとの比較
- 他のDSGEモデル（内閣府、日本銀行モデル）との結果比較

## 🚀 将来の拡張計画

- [ ] 開放経済モデルへの拡張（輸出入、為替レート）
- [ ] 異質的家計の導入（所得分布の考慮）
- [ ] 金融摩擦の追加（銀行部門、信用制約）
- [ ] 人口動態の考慮（少子高齢化の影響）
- [ ] 不確実性・学習の導入
- [ ] リアルタイムデータとの連携

## 📖 参考文献

### 理論的基礎
- Galí, J. (2015). "Monetary Policy, Inflation, and the Business Cycle"
- Woodford, M. (2003). "Interest and Prices"

### 日本経済への応用
- 内閣府経済社会総合研究所 (2018)「マクロ計量モデル（短期日本経済マクロ計量モデル）」
- 日本銀行調査統計局 (2020)「四半期日本経済モデル（Q-JEM）」

### 税制分析
- Trabandt, M. & Uhlig, H. (2011). "The Laffer Curve Revisited"
- Mendoza, E. G., et al. (1994). "Effective Tax Rates in Macroeconomics"

## 📝 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🤝 貢献

プロジェクトへの貢献を歓迎します。以下の手順でご参加ください：

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add some amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📧 連絡先

質問やフィードバックがございましたら、お気軽にお問い合わせください。

---

**注意**: このモデルはシミュレーション・分析目的で開発されており、実際の政策決定においては追加的な検証と専門家による評価が必要です。