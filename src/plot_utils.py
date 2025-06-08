"""
Plotting utilities with Japanese font support
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sys
import os


def setup_japanese_fonts():
    """
    Setup Japanese fonts for matplotlib on different platforms
    """
    # Get available fonts with detailed information
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    available_fonts_set = set(available_fonts)
    
    # Debug: Print some available fonts
    print(f"デバッグ: 利用可能フォント数 {len(available_fonts)}")
    hiragino_fonts = [f for f in available_fonts if 'Hiragino' in f]
    if hiragino_fonts:
        print(f"デバッグ: Hiraginoフォント: {hiragino_fonts[:3]}")
    
    # Define Japanese font candidates (in order of preference)
    japanese_font_candidates = [
        'Hiragino Sans',           # macOS
        'Hiragino Kaku Gothic Pro', # macOS  
        'Yu Gothic',               # Windows/macOS
        'Meiryo',                  # Windows
        'MS Gothic',               # Windows
        'Takao Gothic',            # Linux
        'IPAexGothic',             # Linux
        'Noto Sans CJK JP',        # Cross-platform
        'DejaVu Sans'              # Fallback
    ]
    
    # Find the first available Japanese font
    selected_font = None
    for font in japanese_font_candidates:
        if font in available_fonts_set:
            selected_font = font
            print(f"デバッグ: フォント '{font}' を発見しました")
            break
        else:
            print(f"デバッグ: フォント '{font}' は利用できません")
    
    # Force use Hiragino Sans if on macOS and available
    if not selected_font and 'Hiragino Sans' in available_fonts_set:
        selected_font = 'Hiragino Sans'
        print("デバッグ: Hiragino Sansを強制使用")
    
    if selected_font:
        print(f"✅ 日本語フォント使用: {selected_font}")
        # Set matplotlib font parameters with forced configuration
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans', 'Arial']
        
        # Ensure negative sign displays correctly  
        plt.rcParams['axes.unicode_minus'] = False
        
        # Clear font cache to ensure changes take effect
        fm.fontManager.__init__()
        
        return selected_font
    else:
        print("⚠️ 日本語フォントが見つかりません。デフォルトフォントを使用")
        print("利用可能フォント（先頭10個）:", available_fonts[:10])
        return None


def setup_plotting_style():
    """
    Setup consistent plotting style for the project
    """
    # Setup Japanese fonts first
    font_name = setup_japanese_fonts()
    
    # Set style parameters
    plt.style.use('default')  # Use default instead of seaborn to avoid warnings
    
    # Figure and font settings
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 11
    
    # Grid and spines
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    
    # Colors
    plt.rcParams['axes.prop_cycle'] = plt.cycler(
        'color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                 '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    )
    
    return font_name


def create_comparison_plot(baseline_data, reform_data, variables, 
                          reform_name="Tax Reform", figsize=(12, 8)):
    """
    Create a comparison plot with proper Japanese font support
    """
    n_vars = len(variables)
    n_cols = 2
    n_rows = (n_vars + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, var in enumerate(variables):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Plot data
        if var in baseline_data.columns and var in reform_data.columns:
            ax.plot(baseline_data.index, baseline_data[var], 
                   'b-', label='ベースライン', linewidth=2, alpha=0.8)
            ax.plot(reform_data.index, reform_data[var], 
                   'r--', label='改革後', linewidth=2)
            
            # Add percentage change annotation
            if len(baseline_data) > 0 and len(reform_data) > 0:
                baseline_final = baseline_data[var].iloc[-1]
                reform_final = reform_data[var].iloc[-1]
                if abs(baseline_final) > 1e-10:
                    pct_change = (reform_final - baseline_final) / baseline_final * 100
                    ax.text(0.02, 0.98, f'変化: {pct_change:+.1f}%',
                           transform=ax.transAxes, fontsize=10,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='white', alpha=0.8))
        
        ax.set_title(var, fontweight='bold')
        ax.set_xlabel('期間')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_vars, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.suptitle(f'推移ダイナミクス: {reform_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


def safe_japanese_title(title, fallback_title=None):
    """
    Create a safe title that works with available fonts
    
    Args:
        title: Preferred title (may contain Japanese)
        fallback_title: English fallback title
    
    Returns:
        Safe title string
    """
    try:
        # Test if the title can be rendered
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.set_title(title)
        plt.close(fig)
        return title
    except:
        if fallback_title:
            return fallback_title
        else:
            # Create simple English version
            return title.encode('ascii', errors='ignore').decode('ascii')


def print_font_info():
    """
    Print information about available fonts for debugging
    """
    print("=== Font Information ===")
    print(f"Current font family: {plt.rcParams['font.family']}")
    print(f"Current sans-serif fonts: {plt.rcParams['font.sans-serif'][:3]}")
    print(f"Unicode minus: {plt.rcParams['axes.unicode_minus']}")
    
    # List some Japanese fonts if available
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    japanese_fonts = [f for f in available_fonts if any(
        keyword in f.lower() for keyword in ['hiragino', 'gothic', 'meiryo', 'noto', 'takao']
    )]
    
    if japanese_fonts:
        print(f"Available Japanese fonts: {japanese_fonts[:5]}")
    else:
        print("No obvious Japanese fonts detected")


# Initialize fonts when module is imported
if __name__ != "__main__":
    setup_plotting_style()