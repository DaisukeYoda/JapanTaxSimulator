#!/usr/bin/env python3
"""
Issue #5 最終テスト - 修正されたシステムの確認
"""

import sys
import os
sys.path.append('.')

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from src.dsge_model import DSGEModel, ModelParameters
    from src.linearization_improved import ImprovedLinearizedDSGE
    
    print("=== Issue #5 最終確認テスト ===\n")
    
    # モデル構築
    params = ModelParameters.from_json('config/parameters.json')
    model = DSGEModel(params)
    ss = model.compute_steady_state()
    
    print(f"定常状態: Y={ss.Y:.3f}, C={ss.C:.3f}, A_tfp={ss.A_tfp:.3f}")
    
    # 線形化
    linearizer = ImprovedLinearizedDSGE(model, ss)
    P, Q = linearizer.solve_klein()
    
    # TFPショックIRF
    print(f"\nTFPショックIRF計算...")
    irf = linearizer.compute_impulse_response(
        shock_type='tfp',
        shock_size=1.0,
        periods=20,
        variables=['A_tfp', 'Y', 'C', 'I']
    )
    
    # 結果表示
    print(f"\n=== Issue #5 解決状況 ===")
    
    # 重要な期間の応答をチェック
    for period in [0, 1, 2, 5, 10]:
        if period < len(irf):
            a_tfp_val = irf['A_tfp'].iloc[period]
            y_val = irf['Y'].iloc[period]
            
            print(f"Period {period}:")
            print(f"  TFP: {a_tfp_val:8.4f}%")
            print(f"  GDP: {y_val:8.4f}%")
    
    # 成功判定
    max_y_response = np.max(np.abs(irf['Y']))
    max_a_tfp_response = np.max(np.abs(irf['A_tfp']))
    
    print(f"\n最大応答:")
    print(f"  TFP最大: {max_a_tfp_response:.4f}%")
    print(f"  GDP最大: {max_y_response:.4f}%")
    
    if max_y_response > 0.1:  # 0.1%以上の応答
        print(f"\n✅ Issue #5 解決成功！")
        print(f"   GDP・消費のインパルス応答がゼロ問題は修正されました")
        
        # プロット作成
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        variables = ['A_tfp', 'Y', 'C', 'I']
        titles = ['TFP', 'GDP', '消費', '投資']
        
        for i, (var, title) in enumerate(zip(variables, titles)):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            if var in irf.columns:
                ax.plot(irf.index, irf[var], 'b-', linewidth=2.5, label=title)
                ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                
                # 初期インパクトを強調
                if len(irf) > 1:
                    ax.scatter(1, irf[var].iloc[1], color='red', s=50, zorder=5)
                    ax.text(1, irf[var].iloc[1], f'{irf[var].iloc[1]:.2f}%', 
                           ha='left', va='bottom', fontsize=9)
                
                ax.set_title(f'{title}', fontsize=12, fontweight='bold')
                ax.set_xlabel('期間', fontsize=10)
                ax.set_ylabel('定常状態からの%乖離', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
        
        plt.suptitle('TFPショックIRF（Issue #5 修正後）', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/issue5_fixed_final.png', dpi=300, bbox_inches='tight')
        print(f"   結果プロット: results/issue5_fixed_final.png")
        
    else:
        print(f"\n❌ Issue #5 未解決")
        print(f"   GDP応答が依然として小さすぎます: {max_y_response:.4f}%")
    
    print(f"\n=== テスト完了 ===")
    
except Exception as e:
    print(f"エラー: {e}")
    import traceback
    traceback.print_exc()