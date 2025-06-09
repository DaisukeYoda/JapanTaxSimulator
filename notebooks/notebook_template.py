"""
Notebook Template Generator
Issue #34: 新notebook環境のテンプレート生成

全てのnotebookで一貫した構造を提供するテンプレート生成器
"""

from typing import Dict, List, Optional
import json


class NotebookTemplate:
    """Jupyter Notebook テンプレート生成クラス"""
    
    @staticmethod
    def create_initialization_cells() -> List[Dict]:
        """初期化セル群を生成"""
        return [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# {notebook_title}\n\n",
                    "**目的**: {notebook_purpose}\n\n",
                    "**対象**: {target_audience}\n\n",
                    "**前提知識**: {prerequisites}\n\n",
                    "---\n\n",
                    "このnotebookは**100%実行可能**で、研究整合性を保証します。\n\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 📦 環境初期化\n",
                    "import sys\n",
                    "import os\n",
                    "import warnings\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "\n",
                    "# 共通インフラの読み込み\n",
                    "from notebooks.common import (\n",
                    "    NotebookEnvironment,\n",
                    "    load_baseline_model,\n",
                    "    create_research_simulator,\n",
                    "    validate_research_compliance,\n",
                    "    print_research_disclaimer\n",
                    ")\n",
                    "\n",
                    "# Notebook環境の初期化\n",
                    "NOTEBOOK_NAME = \"{notebook_name}\"\n",
                    "RESEARCH_MODE = {research_mode}  # 研究グレード要求\n",
                    "\n",
                    "env = NotebookEnvironment(NOTEBOOK_NAME, research_mode=RESEARCH_MODE)\n",
                    "env_info = env.setup_environment()\n",
                    "\n",
                    "print(f\"🚀 {NOTEBOOK_NAME} 初期化完了\")\n",
                    "print(f\"研究モード: {'有効' if RESEARCH_MODE else '無効'}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 🎓 ベースラインモデルの読み込み\n",
                    "try:\n",
                    "    model = load_baseline_model()\n",
                    "    \n",
                    "    # パラメータ概要の表示\n",
                    "    params = model.params\n",
                    "    print(\"=== 主要パラメータ ===\")\n",
                    "    print(f\"割引因子 (β): {params.beta}\")\n",
                    "    print(f\"資本分配率 (α): {params.alpha}\")\n",
                    "    print(f\"消費税率 (τc): {params.tau_c:.1%}\")\n",
                    "    print(f\"所得税率 (τl): {params.tau_l:.1%}\")\n",
                    "    print(f\"法人税率 (τf): {params.tau_f:.1%}\")\n",
                    "    \n",
                    "except Exception as e:\n",
                    "    print(f\"❌ モデル読み込み失敗: {e}\")\n",
                    "    raise"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 🔬 研究グレードシミュレータの初期化\n",
                    "if RESEARCH_MODE:\n",
                    "    print(\"🎓 研究グレードシミュレータを初期化中...\")\n",
                    "    \n",
                    "    try:\n",
                    "        simulator, status_info = create_research_simulator(\n",
                    "            model, \n",
                    "            force_research_mode=True,\n",
                    "            use_simple_linearization=False  # 完全Klein線形化\n",
                    "        )\n",
                    "        \n",
                    "        print(f\"シミュレータタイプ: {status_info['simulator_type']}\")\n",
                    "        print(f\"線形化手法: {status_info['linearization_method']}\")\n",
                    "        print(f\"研究整合性: {status_info['research_compliance']}\")\n",
                    "        \n",
                    "        if status_info['warnings']:\n",
                    "            print(f\"⚠️ 警告: {status_info['warnings']}\")\n",
                    "            \n",
                    "    except Exception as e:\n",
                    "        print(f\"❌ シミュレータ初期化失敗: {e}\")\n",
                    "        if RESEARCH_MODE:\n",
                    "            raise\n",
                    "else:\n",
                    "    print(\"📚 教育モードでシミュレータを初期化...\")\n",
                    "    simulator, status_info = create_research_simulator(\n",
                    "        model, \n",
                    "        force_research_mode=False,\n",
                    "        use_simple_linearization=True  # 安定性重視\n",
                    "    )"
                ]
            }
        ]
    
    @staticmethod
    def create_research_validation_cell() -> Dict:
        """研究整合性チェックセル"""
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 🔍 研究整合性の検証\n",
                "validation_result = validate_research_compliance(simulator)\n",
                "\n",
                "print(\"=== 研究整合性チェック ===\")\n",
                "print(f\"研究適合性: {'✅ 適合' if validation_result['is_research_compliant'] else '❌ 不適合'}\")\n",
                "print(f\"シミュレータ: {validation_result['simulator_type']}\")\n",
                "print(f\"DummyState使用リスク: {validation_result['dummy_state_risk']}\")\n",
                "print(f\"線形化手法: {validation_result['linearization_method']}\")\n",
                "\n",
                "if validation_result['warnings']:\n",
                "    print(\"⚠️ 警告:\")\n",
                "    for warning in validation_result['warnings']:\n",
                "        print(f\"  • {warning}\")\n",
                "\n",
                "# 研究使用の免責事項表示\n",
                "if RESEARCH_MODE:\n",
                "    print_research_disclaimer()"
            ]
        }
    
    @staticmethod
    def create_markdown_section(title: str, content: str) -> Dict:
        """マークダウンセクション生成"""
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"## {title}\n\n{content}\n"]
        }
    
    @staticmethod
    def create_error_handling_wrapper(code_content: str, description: str) -> Dict:
        """エラーハンドリング付きコードセル"""
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"# {description}\n",
                "try:\n",
                f"    {code_content}\n",
                "    print(f\"✅ {description}完了\")\n",
                "    \n",
                "except Exception as e:\n",
                "    print(f\"❌ {description}失敗: {e}\")\n",
                "    if RESEARCH_MODE:\n",
                "        print(\"研究モードでのエラーのため実行を停止します\")\n",
                "        raise\n",
                "    else:\n",
                "        print(\"教育モードのため継続します\")"
            ]
        }
    
    @staticmethod
    def create_complete_notebook(
        notebook_title: str,
        notebook_purpose: str,
        target_audience: str,
        prerequisites: str,
        notebook_name: str,
        research_mode: bool = True,
        custom_cells: Optional[List[Dict]] = None
    ) -> Dict:
        """完全なnotebook構造を生成"""
        
        # メタデータ置換用辞書
        replacements = {
            "{notebook_title}": notebook_title,
            "{notebook_purpose}": notebook_purpose,
            "{target_audience}": target_audience,
            "{prerequisites}": prerequisites,
            "{notebook_name}": notebook_name,
            "{research_mode}": str(research_mode)
        }
        
        # 初期化セル群を生成
        init_cells = NotebookTemplate.create_initialization_cells()
        
        # 文字列置換
        for cell in init_cells:
            if cell["cell_type"] == "markdown":
                cell["source"] = [
                    line.format(**replacements) for line in cell["source"]
                ]
            elif cell["cell_type"] == "code":
                cell["source"] = [
                    line.format(**replacements) for line in cell["source"]
                ]
        
        # 研究整合性チェックセル
        validation_cell = NotebookTemplate.create_research_validation_cell()
        
        # 全セルを結合
        all_cells = init_cells + [validation_cell]
        
        # カスタムセルを追加
        if custom_cells:
            all_cells.extend(custom_cells)
        
        # 完全なnotebook構造
        notebook = {
            "cells": all_cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python", 
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {"name": "ipython", "version": 3},
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.0"
                },
                "custom": {
                    "research_mode": research_mode,
                    "creation_tool": "NotebookTemplate",
                    "issue": "Issue #34"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        return notebook


# 事前定義されたnotebook設定
NOTEBOOK_CONFIGS = {
    "dsge_basics": {
        "title": "DSGE モデル入門 - 日本経済の基本構造",
        "purpose": "Dynamic Stochastic General Equilibrium (DSGE) モデルの基本概念を学習し、日本経済の基本構造を理解する",
        "audience": "学部上級生・大学院生・政策担当者",
        "prerequisites": "ミクロ経済学・マクロ経済学の基礎知識",
        "research_mode": False
    },
    
    "tax_fundamentals": {
        "title": "税制政策の基礎 - DSGEによる政策分析入門",
        "purpose": "DSGEモデルを用いた税制政策分析の基礎手法を習得し、日本の税制システムを理解する",
        "audience": "政策担当者・研究者・大学院生",
        "prerequisites": "DSGEモデルの基礎知識・税制の基本概念",
        "research_mode": False
    },
    
    "research_simulation": {
        "title": "研究グレード税制シミュレーション - 学術・政策分析",
        "purpose": "学術研究・政策分析に使用可能な厳密なDSGE税制シミュレーションを実行する",
        "audience": "研究者・政策分析官・博士課程学生",
        "prerequisites": "DSGE理論・数値計算手法・税制政策の高度な知識",
        "research_mode": True
    },
    
    "policy_scenarios": {
        "title": "政策シナリオ分析 - 複数税制改革案の比較評価",
        "purpose": "複数の税制改革シナリオを比較評価し、政策立案に資する分析を提供する",
        "audience": "政策立案者・財務省職員・シンクタンク研究員",
        "prerequisites": "税制政策の実務知識・DSGE分析の基礎",
        "research_mode": True
    }
}