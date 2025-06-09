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
| $\beta$ | 0.99 | 割引因子（四半期） |
| $\sigma_c$ | 1.5 | 異時点間代替の弾力性 |
| $\sigma_l$ | 2.0 | フリッシュ労働供給弾力性 |
| $\alpha$ | 0.33 | 資本分配率 |
| $\theta_p$ | 0.75 | カルボ価格硬直性 |
| $\phi_\pi$ | 1.5 | テイラー則インフレ係数 |
| $\tau_c$ | 0.10 | 消費税率（ベースライン10%） |
| $\tau_l$ | 0.20 | 所得税率（平均） |

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
├── pyproject.toml                     # uv依存管理・プロジェクト設定
├── CLAUDE.md                          # Claude Code開発ガイド
├── config/
│   └── parameters.json               # モデルパラメータ設定
├── src/                              # 🆕 モジュラーアーキテクチャ
│   ├── dsge_model.py                 # 基本DSGEモデル実装
│   ├── linearization.py             # 線形化モジュール
│   ├── linearization_improved.py    # Klein線形化（研究用）
│   ├── tax_simulator.py             # 🔄 後方互換性ファサード
│   ├── simulation/                  # 🆕 シミュレーションエンジン
│   │   ├── base_simulator.py        #   - 基本シミュレーション基盤
│   │   └── enhanced_simulator.py    #   - 高度なDSGEシミュレーション
│   ├── analysis/                    # 🆕 経済分析モジュール
│   │   ├── welfare_analysis.py      #   - 厚生分析（複数手法対応）
│   │   └── fiscal_impact.py         #   - 財政インパクト分析
│   ├── utils_new/                   # 🆕 強化されたユーティリティ
│   │   ├── reform_definitions.py    #   - 税制改革定義・検証
│   │   └── result_containers.py     #   - 結果データ管理
│   └── models/                      # DSGEモデル実装
│       └── simple_dsge.py           #   - 教育用簡単モデル
├── notebooks/                        # インタラクティブ分析
│   ├── tax_simulation_demo.ipynb    # 基本デモ
│   ├── advanced_tax_simulation_demo.ipynb  # 高度な分析デモ
│   ├── interactive_tax_analysis.ipynb      # インタラクティブ分析
│   └── empirical_validation.ipynb   # 実証検証
├── tests/                           # 包括的テストスイート
│   ├── unit/                        # ユニットテスト
│   └── integration/                 # 統合テスト
├── scripts/                         # 分析・検証スクリプト
│   └── validation/                  # モデル検証ツール
├── docs/                           # 🆕 体系化されたドキュメント
│   ├── REFACTORING_COMPLETION_SUMMARY.md  # リファクタリング完了報告
│   ├── development/                 # 開発者向けドキュメント
│   │   ├── setup.md                 # セットアップガイド  
│   │   └── TAX_REFORM_TROUBLESHOOTING.md  # トラブルシューティング
│   └── technical/                   # 技術ドキュメント
│       ├── MODULAR_ARCHITECTURE_GUIDE.md # アーキテクチャガイド
│       ├── RESEARCH_INTEGRITY_STATUS.md  # 研究品質ステータス
│       ├── TECHNICAL_DOCS.md        # 技術仕様書
│       └── LINEARIZATION_METHOD_GUIDE.md # 線形化手法ガイド
├── data/                           # データファイル
├── results/                        # シミュレーション結果
└── quick_check.py                  # クイック動作確認
```

### 🏗️ 新しいモジュラーアーキテクチャ（2025年6月版）

**🎯 主要改善点:**
- **分離された関心事**: シミュレーション、分析、ユーティリティが明確に分離
- **研究グレード品質**: 複数の手法、明示的な仮定、学術的検証機能
- **100%後方互換性**: 既存のコードは変更なしで動作
- **包括的ドキュメント**: 技術仕様から開発ガイドまで完備

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

#### 🔄 既存ユーザー向け（後方互換性）

```python
# 既存のコードは変更なしで動作します
from src.dsge_model import DSGEModel, ModelParameters
from src.tax_simulator import EnhancedTaxSimulator, TaxReform

# モデルの初期化
params = ModelParameters.from_json('config/parameters.json')
model = DSGEModel(params)

# シミュレーターの作成（既存インターフェース）
simulator = EnhancedTaxSimulator(model)

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

#### 🆕 新しいモジュラーAPI（推奨）

```python
# より精密な制御とより優れた分析機能
from src.dsge_model import DSGEModel, ModelParameters
from src.simulation.enhanced_simulator import EnhancedSimulationEngine, LinearizationConfig
from src.analysis.welfare_analysis import WelfareAnalyzer, WelfareConfig
from src.analysis.fiscal_impact import FiscalAnalyzer, FiscalConfig
from src.utils_new.reform_definitions import TaxReform

# モデル初期化
params = ModelParameters.from_json('config/parameters.json')
model = DSGEModel(params)

# 研究グレードシミュレーションエンジン
sim_engine = EnhancedSimulationEngine(
    baseline_model=model,
    linearization_config=LinearizationConfig(
        method='klein',  # 研究用Klein線形化
        validate_bk_conditions=True
    ),
    research_mode=True
)

# 高度な厚生分析
welfare_analyzer = WelfareAnalyzer(
    config=WelfareConfig(
        methodology='consumption_equivalent',
        include_uncertainty=True,  # 信頼区間付き
        confidence_level=0.95
    )
)

# 財政インパクト分析
fiscal_analyzer = FiscalAnalyzer(
    config=FiscalConfig(
        include_behavioral_responses=True,
        include_general_equilibrium=True
    )
)

# 税制改革シミュレーション
reform = TaxReform(
    name="包括的税制改革",
    tau_c=0.12,    # 消費税12%
    tau_l=0.18,    # 所得税18%
    implementation='phased',
    phase_in_periods=8
)

# 実行
sim_results = sim_engine.simulate_reform(reform, periods=40)
welfare_results = welfare_analyzer.analyze_welfare_impact(
    sim_results.baseline_path, sim_results.reform_path
)
fiscal_results = fiscal_analyzer.analyze_fiscal_impact(
    sim_results.reform_path, model.parameters, sim_results.baseline_path
)

# 包括的な結果分析
print(f"厚生変化: {welfare_results.consumption_equivalent:.2%}")
print(f"財政インパクト: {fiscal_results.present_value_impact:.2f}")
```

#### 🎓 教育・デモ用途

```python
# シンプルな線形化を使用（安定、理解しやすい）
from src.tax_simulator import EnhancedTaxSimulator

simulator = EnhancedTaxSimulator(
    model, 
    use_simple_linearization=True  # 教育用
)
results = simulator.simulate_reform(reform)
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
| GDP | $-1.2\%$ | $-0.8\%$ | $-0.5\%$ |
| 消費 | $-2.1\%$ | $-1.4\%$ | $-1.0\%$ |
| 投資 | $-0.8\%$ | $-0.4\%$ | $-0.2\%$ |
| 物価水準 | $+4.5\%$ | $+4.8\%$ | $+5.0\%$ |
| 総税収 | $+2.8\%$ | $+3.2\%$ | $+3.5\%$ |

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

```math
\max E_0 \sum_{t=0}^{\infty} \beta^t \left[ \log(C_t - hC_{t-1}) - \frac{\chi N_t^{1+1/\sigma_l}}{1+1/\sigma_l} \right]
```

制約条件：
```math
(1+\tau_c)C_t + I_t + B_t \leq (1-\tau_l)W_tN_t + (1-\tau_k)R_tK_t + \frac{B_{t-1}}{\pi_t} + T_t
```

#### 企業の利潤最大化

```math
\max E_0 \sum_{t=0}^{\infty} \Lambda_t \left[ (1-\tau_f)(P_tY_t - W_tN_t - R_tK_t) - \frac{\psi}{2}\left(\frac{I_t}{K_{t-1}} - \delta\right)^2 K_{t-1} \right]
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

## 🆕 2025年版の新機能・改善点

### ✅ 完成済み（2025年6月）

#### 🏗️ **モジュラーアーキテクチャ**
- **研究グレード品質**: 複数の厚生分析手法、明示的な仮定、学術的検証
- **分離された関心事**: シミュレーション、分析、ユーティリティモジュール
- **100%後方互換性**: 既存コードは変更なしで動作
- **包括的ドキュメント**: 技術仕様書から開発ガイドまで

#### 🧮 **高度な線形化手法**
- **Klein法（研究用）**: DSGE理論に基づく厳密な解法
- **簡単線形化（教育用）**: 安定で理解しやすい近似解法
- **自動選択**: 用途に応じた適切な手法の選択

#### 📊 **強化された分析機能**
- **複数の厚生分析手法**: 消費等価変分、Lucas厚生指標
- **信頼区間付き結果**: ブートストラップによる不確実性定量化
- **包括的財政分析**: 税収・支出・債務動学の詳細分析
- **比較分析**: 複数シナリオの自動比較機能

#### 🔬 **研究整合性システム**
- **明示的な仮定**: すべての仮定が明確に文書化
- **検証警告**: 研究用途での注意事項を自動表示
- **パラメータ検証**: 経済的妥当性のチェック機能
- **再現可能性**: 結果の完全な再現性を保証

### 🚀 将来の拡張計画

#### Phase 1: 国際化・開放経済 (2025年後半)
- [ ] 開放経済モデルへの拡張（輸出入、為替レート）
- [ ] 国際的な税制調整の分析
- [ ] 為替レート政策との相互作用

#### Phase 2: 異質性の導入 (2026年前半)
- [ ] 異質的家計の導入（所得分布の考慮）
- [ ] 世代重複モデル（OLGモデル）
- [ ] 企業規模の異質性

#### Phase 3: 金融セクター (2026年後半)
- [ ] 金融摩擦の追加（銀行部門、信用制約）
- [ ] 金融政策と財政政策の相互作用
- [ ] 資産価格動学

#### Phase 4: 人口動態・長期分析 (2027年)
- [ ] 人口動態の考慮（少子高齢化の影響）
- [ ] 社会保障制度との統合
- [ ] 長期財政持続可能性分析

#### Phase 5: 高度な不確実性 (2027年後半)
- [ ] 不確実性・学習の導入
- [ ] リアルタイムデータとの連携
- [ ] 機械学習による政策最適化

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

プロジェクトへの貢献を歓迎します！新しいモジュラーアーキテクチャにより、より多くの貢献の機会があります。

### 🚀 貢献の種類

#### 📊 **分析手法の追加**
- 新しい厚生分析手法の実装
- 追加的な財政分析機能
- 新しい経済指標の計算

#### 🧮 **数値手法の改良**
- より高速な線形化アルゴリズム
- 並列計算の実装
- 数値安定性の向上

#### 📚 **教育・デモ機能**
- インタラクティブな可視化
- 教育用の簡単化されたモデル
- より良いドキュメント

#### 🔬 **研究品質の向上**
- 実証的な検証
- パラメータキャリブレーションの改良
- 国際比較機能

### 🛠️ 開発手順

1. **リポジトリをフォーク**
2. **機能ブランチを作成**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **開発環境のセットアップ**
   ```bash
   uv sync
   uv run python quick_check.py  # 動作確認
   ```
4. **モジュラーアーキテクチャに従って開発**
   - `src/simulation/`: シミュレーション機能
   - `src/analysis/`: 分析機能
   - `src/utils_new/`: ユーティリティ機能
5. **テストを追加・実行**
   ```bash
   uv run pytest tests/
   ```
6. **ドキュメント更新**
   - コード内のdocstring
   - 必要に応じて技術ドキュメント更新
7. **変更をコミット**
   ```bash
   git commit -m 'Add: 新しい厚生分析手法を追加'
   ```
8. **プルリクエストを作成**

### 📋 コーディング規約

- **研究整合性**: 明示的な仮定、適切な検証
- **モジュラー設計**: 明確な責任分離
- **後方互換性**: 既存APIの維持
- **包括的テスト**: 新機能には必ずテスト追加

## 📧 連絡先・サポート

### 🐛 問題報告・機能要望
- **GitHub Issues**: [Issues ページ](https://github.com/DaisukeYoda/JapanTaxSimulator/issues)
- **プルリクエスト**: [PR ページ](https://github.com/DaisukeYoda/JapanTaxSimulator/pulls)

### 📚 ドキュメント・使用法
- **技術ドキュメント**: `docs/technical/` 参照
- **開発者ガイド**: `docs/development/` 参照  
- **モジュラーアーキテクチャ**: `docs/technical/MODULAR_ARCHITECTURE_GUIDE.md`

### 🎓 学術利用・研究協力
研究目的での利用や共同研究についてはGitHub Issuesでお気軽にご相談ください。

---

**📈 パフォーマンス**: このプロジェクトは研究品質とパフォーマンスの両立を目指しています  
**🔬 品質保証**: 学術研究での利用を前提とした厳格な品質管理  
**🌐 オープンソース**: MIT Licenseによる自由な利用・改変

---

**注意**: このモデルはシミュレーション・分析目的で開発されており、実際の政策決定においては追加的な検証と専門家による評価が必要です。