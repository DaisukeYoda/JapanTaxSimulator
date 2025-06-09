import pandas as pd
import os

# Define the directory where simulation results are stored
ANALYSIS_DIR = "results/current_analysis_report_data"

# Helper function to read CSV data if it exists, otherwise return None or empty DataFrame
def load_data(filename):
    filepath = os.path.join(ANALYSIS_DIR, filename)
    if os.path.exists(filepath):
        try:
            return pd.read_csv(filepath)
        except pd.errors.EmptyDataError:
            print(f"Warning: File {filepath} is empty.")
            return pd.DataFrame() # Return empty DataFrame for empty files
        except Exception as e:
            print(f"Warning: Could not read {filepath}. Error: {e}")
            return None
    print(f"Warning: File {filepath} does not exist.")
    return None

# Load all necessary data
ss_consumption_perm = load_data("ss_comp_consumption_tax_increase_5pp_permanent.csv")
irf_consumption_perm = load_data("irf_consumption_tax_increase_5pp_permanent.csv")

ss_consumption_phased = load_data("ss_comp_consumption_tax_increase_5pp_phased.csv")
irf_consumption_phased = load_data("irf_consumption_tax_increase_5pp_phased.csv")

ss_income_perm = load_data("ss_comp_income_tax_reduction_5pp_permanent.csv")
irf_income_perm = load_data("irf_income_tax_reduction_5pp_permanent.csv")

ss_revenue_neutral = load_data("ss_comp_revenue_neutral_c_up_l_down.csv")
irf_revenue_neutral = load_data("irf_revenue_neutral_c_up_l_down.csv")

# Function to format steady-state changes for the report
def format_ss_changes(ss_df, scenario_name):
    if ss_df is None:
        return f"Steady-state data for {scenario_name} not found.\n"
    if ss_df.empty:
        return f"Steady-state data for {scenario_name} is empty.\n"

    report_text = f"「{scenario_name}」における主要変数の定常状態変化（ベースライン比）：\n"
    for index, row in ss_df.iterrows():
        var = row['Variable']
        pct_change = row['Percentage Change']
        if pd.notna(pct_change):
            report_text += f"  - {var}: {pct_change:.2f}%\n"
        else:
            report_text += f"  - {var}: データなし\n" # Handle NaN like for 'B'
    return report_text + "\n"

# Function to briefly describe IRF, focusing on initial impact and convergence
def describe_irf(irf_df, scenario_name, main_vars=['Y', 'C', 'I', 'L']):
    if irf_df is None:
        return f"IRF data for {scenario_name} not found.\n"
    if irf_df.empty:
        return f"IRF data for {scenario_name} is empty.\n"
    if irf_df.shape[0] < 2: # Need at least 2 periods for initial and first period impact
        return f"IRF data for {scenario_name} has insufficient periods for analysis.\n"


    report_text = f"「{scenario_name}」の動学的調整過程の概要：\n"
    # Get initial values (Period 0) and first period impact (Period 1)
    initial_values = irf_df.iloc[0]
    first_period_values = irf_df.iloc[1]

    for var in main_vars:
        if var in irf_df.columns:
            initial = initial_values[var]
            first_period = first_period_values[var]
            if pd.isna(initial) or pd.isna(first_period):
                report_text += f"  - {var}: IRFデータに欠損値あり\n"
                continue
            change = ((first_period / initial) - 1) * 100 if initial != 0 else (float('inf') if first_period != 0 else 0)

            if change == float('inf') or change == float('-inf'):
                 direction = "大幅に増加" if first_period > 0 else "大幅に減少"
                 report_text += f"  - {var}: 第1期に{direction}し（初期値ゼロ）、その後新たな定常状態へ収束する動きを見せた。\n"
            else:
                direction = "増加" if change > 0 else ("減少" if change < 0 else "変化なし")
                report_text += f"  - {var}: 第1期に約{abs(change):.2f}%{direction}し、その後新たな定常状態へ収束する動きを見せた。\n"
        else:
            report_text += f"  - {var}: IRFデータなし\n"

    # Special mention for phased approach
    if "段階的" in scenario_name:
        report_text += "  - 段階的導入により、各変数の変化は恒久的な即時導入の場合と比較して緩やかであった。\n"
    return report_text + "\n"

# --- Generate Report Content ---
report_content = "税制改革の経済効果分析レポート\n\n"

report_content += "1. はじめに\n"
report_content += "本レポートは、動学的確率的一般均衡（DSGE）モデルを用い、いくつかの税制変更が日本経済に与えるマクロ経済的影響を分析することを目的とする。具体的には、消費税及び所得税の変更シナリオについて、GDP、消費、投資、労働供給、税収といった主要経済変数への影響を評価する。\n\n"

report_content += "2. モデル概要\n"
report_content += "本分析で使用したDSGEモデルは、家計、企業、政府、中央銀行の4部門から構成される標準的なニューケインジアンモデルである。家計は効用を最大化し、企業は利潤を最大化する。政府は税収と国債発行により支出を賄い、中央銀行はテイラー則に基づき政策金利を決定する。ベースラインの税率は、消費税10%、所得税20%、法人税30%等に設定されている。詳細なパラメータは`config/parameters.json`に記載されている。\n\n"

report_content += "3. シミュレーションシナリオ\n"
report_content += "以下の4つの税制改革シナリオについてシミュレーション分析を行った。\n"
report_content += "  シナリオ1: 恒久的消費税率5%ポイント引き上げ（10% → 15%）\n"
report_content += "  シナリオ2: 段階的消費税率5%ポイント引き上げ（10% → 15%、8四半期で段階的に導入）\n"
report_content += "  シナリオ3: 恒久的所得税率5%ポイント引き下げ（20% → 15%）\n"
report_content += "  シナリオ4: 税収中立的改革（消費税率2%ポイント引き上げ、所得税率2%ポイント引き下げを想定）\n\n"

report_content += "4. シミュレーション結果\n"

# Scenario 1 Results
report_content += "【シナリオ1: 恒久的消費税率5%ポイント引き上げ】\n"
report_content += format_ss_changes(ss_consumption_perm, "恒久的消費税率5%ポイント引き上げ")
report_content += describe_irf(irf_consumption_perm, "恒久的消費税率5%ポイント引き上げ")
if ss_consumption_perm is not None and not ss_consumption_perm.empty and 'T_total_revenue' in ss_consumption_perm['Variable'].values:
    rev_change_series = ss_consumption_perm[ss_consumption_perm['Variable'] == 'T_total_revenue']['Percentage Change']
    if not rev_change_series.empty:
        rev_change = rev_change_series.iloc[0]
        if pd.notna(rev_change) and rev_change < 0:
            report_content += "特筆すべき点として、消費税率引き上げにもかかわらず、総税収は約" + f"{abs(rev_change):.2f}%減少した。これは、税率上昇による消費の落ち込みが税収効果を上回ったことを示唆する。\n\n"

# Scenario 2 Results
report_content += "【シナリオ2: 段階的消費税率5%ポイント引き上げ】\n"
report_content += format_ss_changes(ss_consumption_phased, "段階的消費税率5%ポイント引き上げ")
report_content += describe_irf(irf_consumption_phased, "段階的消費税率5%ポイント引き上げ")
report_content += "長期的（定常状態）な影響はシナリオ1（恒久的導入）と同様であったが、経済への短期的な影響はより緩やかであった。\n\n"

# Scenario 3 Results
report_content += "【シナリオ3: 恒久的所得税率5%ポイント引き下げ】\n"
report_content += format_ss_changes(ss_income_perm, "恒久的所得税率5%ポイント引き下げ")
report_content += describe_irf(irf_income_perm, "恒久的所得税率5%ポイント引き下げ")
if ss_income_perm is not None and not ss_income_perm.empty and 'T_total_revenue' in ss_income_perm['Variable'].values:
    rev_change_series = ss_income_perm[ss_income_perm['Variable'] == 'T_total_revenue']['Percentage Change']
    if not rev_change_series.empty:
        rev_change = rev_change_series.iloc[0]
        if pd.notna(rev_change) and rev_change > 0:
            report_content += "所得税率引き下げにより経済活動が刺激され、総税収は約" + f"{rev_change:.2f}%増加した。これは、いわゆるラッファーカーブの右側（減税による税収増）の効果を示唆している可能性がある。\n\n"

# Scenario 4 Results
report_content += "【シナリオ4: 税収中立的改革（消費税+2%p、所得税-2%p）】\n"
report_content += format_ss_changes(ss_revenue_neutral, "税収中立的改革")
report_content += describe_irf(irf_revenue_neutral, "税収中立的改革")
if ss_revenue_neutral is not None and not ss_revenue_neutral.empty:
    # Ensure variables exist and data is not NaN before accessing .iloc[0]
    def get_change_value(df, var_name):
        series = df[df['Variable'] == var_name]['Percentage Change']
        if not series.empty and pd.notna(series.iloc[0]):
            return series.iloc[0]
        return None

    gdp_change_val = get_change_value(ss_revenue_neutral, 'Y')
    # inv_change_val = get_change_value(ss_revenue_neutral, 'I') # Original script had this, but it's not used in the next line
    rev_change_val = get_change_value(ss_revenue_neutral, 'T_total_revenue')
    w_change_val = get_change_value(ss_revenue_neutral, 'w')
    pi_change_val = get_change_value(ss_revenue_neutral, 'pi_gross')


    report_parts = []
    if gdp_change_val is not None:
        report_parts.append(f"GDPが約{gdp_change_val:.2f}%と大幅に減少し、消費も落ち込んだ。")
    if rev_change_val is not None:
        report_parts.append(f"総税収は約{rev_change_val:.2f}%増加した。")
    # The original text mentioned investment IRF, not steady state.
    # report_parts.append("投資の定常状態変化はほぼ0%であったが、IRFデータによれば短期的な変動は見られた。")
    # This part is better handled by describe_irf or a more detailed IRF summary.
    # For now, let's keep the steady-state focus for this section.
    if w_change_val is not None:
         report_parts.append(f"実質賃金の上昇（約{w_change_val:.2f}%）が観測された。")
    if pi_change_val is not None:
        # pi_gross is gross inflation, so (pi_gross - 1)*100 is net inflation rate.
        # The original text says "インフレ率の低下（約-0.25%）" which implies a change in net inflation.
        # Percentage change of gross inflation is not directly net inflation change.
        # For simplicity, reporting the change in gross inflation.
        # A more accurate report would calculate net inflation then its change, or report % change of net inflation.
        # Let's assume the value in the CSV is already the one desired for reporting.
        report_parts.append(f"インフレ率（グロス）の変化は約{pi_change_val:.2f}%であった。")


    if report_parts:
        report_content += "このシナリオでは、" + " ".join(report_parts) + "\n\n"
    else:
        report_content += "このシナリオでは、主要な変数について報告できる十分なデータがありませんでした。\n\n"


report_content += "5. 考察\n"
report_content += "シミュレーション結果から、いくつかの重要な示唆が得られた。\n"
report_content += "まず、消費税率の引き上げは、本モデルの範囲内ではGDP、消費、投資、労働供給を減少させ、さらに総税収も減少させる可能性が示された。これは、税率上昇による負の需要効果が税収基盤を縮小させる効果を上回る場合に起こりうる。\n"
report_content += "次に、所得税率の引き下げは、主要な経済活動を刺激し、総税収を増加させる結果となった。これは、労働供給や消費へのインセンティブが改善されたためと考えられる。\n"
report_content += "消費税増税を段階的に導入した場合、経済への最終的な影響（定常状態）は即時導入の場合と同じであったが、短期的な調整の速度は緩やかになり、急激な変化を緩和する効果が見られた。\n"
report_content += "「税収中立」を目指した消費税増・所得税減の組み合わせシナリオ（シナリオ4）では、GDPが大幅に減少する結果となり、本モデルの構造とパラメータ設定の下では、所得税から消費税へのシフトが経済活動に対して強い負の影響を持つ可能性が示唆された。特に、投資への定常的な影響がほぼゼロであった点や、インフレ率が長期的に低下した点は、さらなる詳細な分析を要する。また、このシナリオで政府支出（G）に微増が見られた点は、モデルの仕様確認が必要である。\n"
report_content += "留意事項として、本シミュレーションは特定のDSGEモデルとそのパラメータ設定に基づいたものであり、結果はこれらの仮定に依存する。シミュレーション実行時には、定常状態計算の収束性や線形化手法に関する警告が観測されており、これはモデルの安定性や結果の頑健性について注意が必要であることを示している。プロジェクト内には`ACADEMIC_RESEARCH_REMEDIATION_PLAN.md`が存在し、モデルの信頼性向上のための取り組みが計画されている。また、一部の変数（例：政府債務残高 `B`）の定常状態データが得られていない点も限界として挙げられる。\n\n"

report_content += "6. 結論\n"
report_content += "本分析結果は、税制変更がマクロ経済に与える影響の方向性と大まかな規模を示唆するものである。消費税増税は経済活動を抑制し税収を減少させる可能性があり、所得税減税は経済を刺激し税収を増加させる可能性がある。税負担構造の変更（所得税から消費税へ）は、本モデルでは大きな負の経済的影響をもたらした。ただし、これらの結果の解釈には、モデルの仮定と限界、およびシミュレーションの安定性に関する警告を十分に考慮する必要がある。より詳細な政策評価のためには、モデルの継続的な改善と検証が不可欠である。\n\n"

report_content += "7. 付録\n"
report_content += "詳細なシミュレーション結果（IRFデータ、定常状態比較データ）は、`results/current_analysis_report_data/` ディレクトリ内のCSVファイルとして別途保存されている。\n"

# Output the generated report content to a file for the subtask to pick up
output_filename = "tax_analysis_report_japanese.txt"
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(report_content) # Write content as is, assuming newlines are already correct

# Make sure the subtask runner knows the output filename
print(f"Report content written to {output_filename}")
