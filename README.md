# 日本税制シミュレーター (Japan Tax Simulator)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

日本経済の税制変更がマクロ経済に与える影響を分析するための動学的確率的一般均衡（DSGE）モデル

[English](README_EN.md) | 日本語

## 🚀 クイックスタート

```bash
# リポジトリをクローン
git clone https://github.com/DaisukeYoda/JapanTaxSimulator.git
cd JapanTaxSimulator

# 依存関係をインストール（uvが推奨）
uv sync

# 動作確認
uv run python quick_check.py
```

### 基本的な使用例

```python
from src.dsge_model import DSGEModel, ModelParameters
from src.tax_simulator import EnhancedTaxSimulator, TaxReform

# モデル初期化
params = ModelParameters.from_json('config/parameters.json')
model = DSGEModel(params)
model.steady_state = model.compute_steady_state()

# 消費税1%増税をシミュレーション
simulator = EnhancedTaxSimulator(model)
reform = TaxReform(name="消費税1%増税", tau_c=0.11)
results = simulator.simulate_reform(reform, periods=40)
print(f"厚生変化: {results.welfare_change:.2%}")
```

## 📖 主な機能

- **4つの税制分析**: 消費税、所得税、資本所得税、法人税
- **動学的シミュレーション**: 短期・長期の経済影響を分析
- **厚生分析**: 政策変更による社会厚生の変化を定量評価
- **研究グレードの精度**: 学術研究・政策分析に使用可能

## 📚 ドキュメント

**包括的なガイド**
- **[docs/USER_GUIDE.md](docs/USER_GUIDE.md)** - 詳細な使用方法とAPIリファレンス
- **[docs/EXAMPLES.md](docs/EXAMPLES.md)** - 実用的なコード例とシナリオ分析

**専門分野別**
- **[docs/development/](docs/development/)** - 開発者向け情報
- **[docs/technical/](docs/technical/)** - 技術仕様と理論的背景
- **[docs/research/](docs/research/)** - 政策研究文書
- **[docs/planning/](docs/planning/)** - 将来開発計画

## 📋 ドキュメント一覧

### メインドキュメント
- **[docs/USER_GUIDE.md](docs/USER_GUIDE.md)** - DSGEモデルの包括的ユーザーガイド、税制政策分析の完全な使用方法
- **[docs/EXAMPLES.md](docs/EXAMPLES.md)** - 実際にテストされたAPIコード例集、クイックスタートから高度なシミュレーションまで
- **[docs/REFACTORING_SUMMARY.md](docs/REFACTORING_SUMMARY.md)** - モジュラーアーキテクチャ実装完了サマリー（2025年6月）

### 開発者向け
- **[docs/development/setup.md](docs/development/setup.md)** - uv依存関係管理ツールを使用した開発環境セットアップガイド
- **[docs/development/TAX_REFORM_TROUBLESHOOTING.md](docs/development/TAX_REFORM_TROUBLESHOOTING.md)** - 税制改革シミュレーション実行時の問題解決ガイド

### 技術仕様
- **[docs/technical/LINEARIZATION_METHOD_GUIDE.md](docs/technical/LINEARIZATION_METHOD_GUIDE.md)** - 簡略化vs完全線形化（Klein解法）の選択ガイド（Issue #30対応）
- **[docs/technical/LINEARIZATION_RANK_DEFICIENCY_ISSUE.md](docs/technical/LINEARIZATION_RANK_DEFICIENCY_ISSUE.md)** - DSGEモデル線形化のランク不足問題（rank 5/27）の技術解説
- **[docs/technical/MODULAR_ARCHITECTURE_GUIDE.md](docs/technical/MODULAR_ARCHITECTURE_GUIDE.md)** - 新しいコンポーネントベース設計とクリーンアーキテクチャ仕様
- **[docs/technical/RESEARCH_INTEGRITY_STATUS.md](docs/technical/RESEARCH_INTEGRITY_STATUS.md)** - リファクタリング後の研究整合性改善レポート
- **[docs/technical/TECHNICAL_DOCS.md](docs/technical/TECHNICAL_DOCS.md)** - DSGEモデルの数学的定式化と4部門の理論的基盤
- **[docs/technical/international_social_security_models.md](docs/technical/international_social_security_models.md)** - 海外経済モデルにおける社会保障制度の扱い比較

### 政策研究
- **[docs/research/README.md](docs/research/README.md)** - 政策研究文書の索引、財政政策透明性向上の理論的・実践的基盤
- **[docs/research/civic_fiscal_analysis_examples.md](docs/research/civic_fiscal_analysis_examples.md)** - 民間IFI的機能の国内外事例（Tax-Calculator、OpenFisca、PolicyEngine等）
- **[docs/research/independent_fiscal_institutions.md](docs/research/independent_fiscal_institutions.md)** - 独立財政機関（IFI）の解説、日本の現状とOECD設計原則
- **[docs/research/mof_ifi_politics.md](docs/research/mof_ifi_politics.md)** - 財務省と独立財政機関の政治的関係分析
- **[docs/research/open_source_ifi_model_proposal.md](docs/research/open_source_ifi_model_proposal.md)** - IFIオープンソースモデル構想、open-ifi-japanプロジェクト設計
- **[docs/research/policy_proposal_ifi_japan.md](docs/research/policy_proposal_ifi_japan.md)** - 日本における独立財政機関設置の段階的政策提言とロードマップ
- **[docs/research/transparency_analysis_methodology.md](docs/research/transparency_analysis_methodology.md)** - 政策透明性と政治参加の実証分析手法

### 将来計画
- **[docs/planning/social_security_integration_plan.md](docs/planning/social_security_integration_plan.md)** - 社会保障システム統合の段階的実装ロードマップ

## 🔧 モデル仕様

**標準的なDSGEモデル**（家計・企業・政府・中央銀行の4部門）
- 4つの税制（消費税、所得税、資本所得税、法人税）
- 動学的移行経路と厚生分析機能
- 日本経済データに基づくキャリブレーション

## 💻 インストール

**環境要件**: Python 3.11+

```bash
# uvをインストール（推奨）
brew install uv  # macOS
# または: curl -LsSf https://astral.sh/uv/install.sh | sh

# プロジェクトセットアップ
git clone https://github.com/DaisukeYoda/JapanTaxSimulator.git
cd JapanTaxSimulator
uv sync

# Jupyter Notebookでのデモを実行
uv run jupyter notebook notebooks/
```

## 🤝 コントリビューションとサポート

**質問・バグ報告**: [GitHub Issues](https://github.com/DaisukeYoda/JapanTaxSimulator/issues)
**ライセンス**: [MIT License](LICENSE)

---

このモデルは学術研究・政策分析目的で開発されています。実際の政策決定には専門家による検証が必要です。