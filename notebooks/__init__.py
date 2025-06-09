"""
Notebooks package for Japan Tax Simulator
Issue #34: 教育・研究・政策分析用の統合的学習環境構築
"""

# パッケージ情報
__version__ = "1.0.0"
__author__ = "JapanTaxSimulator Team"
__description__ = "Interactive notebooks for DSGE tax policy analysis"

# 主要クラスのインポート
from .common import (
    NotebookEnvironment,
    load_baseline_model,
    create_research_simulator,
    validate_research_compliance,
    safe_simulation_wrapper,
    create_comparison_dataframe,
    plot_scenario_comparison,
    print_research_disclaimer,
    SAMPLE_TAX_SCENARIOS,
    VARIABLE_NAMES_JP
)

from .notebook_template import (
    NotebookTemplate,
    NOTEBOOK_CONFIGS
)

__all__ = [
    'NotebookEnvironment',
    'load_baseline_model', 
    'create_research_simulator',
    'validate_research_compliance',
    'safe_simulation_wrapper',
    'create_comparison_dataframe',
    'plot_scenario_comparison',
    'print_research_disclaimer',
    'NotebookTemplate',
    'SAMPLE_TAX_SCENARIOS',
    'VARIABLE_NAMES_JP',
    'NOTEBOOK_CONFIGS'
]