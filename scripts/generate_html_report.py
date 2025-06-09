import os
import pandas as pd

TEXT_REPORT_PATH = "results/research_report_tax_analysis_effects_jp.txt"
SIMULATION_DATA_DIR = "results/current_analysis_report_data"
HTML_OUTPUT_FILENAME = "tax_analysis_report_jp.html"

def text_to_html_list(text_block):
    items = text_block.strip().split('\n')
    html_list = "<ul>\n"
    for item in items:
        item = item.strip()
        if item.startswith("シナリオ") and ":" in item:
            item_content = item.split(":", 1)[1].strip()
            html_list += f"  <li>{item_content}</li>\n"
        elif item.startswith("- "):
            item_content = item.replace("- ", "").strip()
            html_list += f"  <li>{item_content}</li>\n"
        elif item: # Catch-all for other non-empty lines
             html_list += f"  <li>{item}</li>\n"
    html_list += "</ul>\n"
    return html_list

def create_ss_table_html(scenario_key_name, table_title):
    csv_path = os.path.join(SIMULATION_DATA_DIR, f"ss_comp_{scenario_key_name}.csv")
    if not os.path.exists(csv_path):
        return f"<p>定常状態変化のデータ ({scenario_key_name}.csv) が見つかりませんでした。</p>\n"
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return f"<p>定常状態変化のデータ ({scenario_key_name}.csv) は空です。</p>\n"

        table_html = f"<h4>{table_title} - 定常状態変化</h4>\n"
        table_html += "<table border='1' style='border-collapse: collapse; width: auto;'>\n" # Added style
        table_html += "<thead><tr><th style='text-align: left; padding: 5px;'>変数</th><th style='text-align: right; padding: 5px;'>ベースライン比 (%)</th></tr></thead>\n<tbody>\n" # Added style
        for _, row in df.iterrows():
            var = row['Variable']
            pct_change = row['Percentage Change']
            pct_change_str = f"{pct_change:.2f}" if pd.notna(pct_change) else "データなし"
            # Display only key variables as in the original script's intent
            if var in ['Y', 'C', 'I', 'L', 'T_total_revenue', 'w', 'pi_gross', 'G', 'B']: # Added more key vars based on text report
                 table_html += f"  <tr><td style='padding: 5px;'>{var}</td><td style='text-align: right; padding: 5px;'>{pct_change_str}</td></tr>\n" # Added style
        table_html += "</tbody>\n</table>\n"
        return table_html
    except Exception as e:
        return f"<p>定常状態変化のテーブル作成中にエラー ({scenario_key_name}.csv): {e}</p>\n"

plot_files = {
    "シナリオ1": "irf_plot_consumption_tax_increase_5pp_permanent.png",
    "シナリオ2": "irf_plot_consumption_tax_increase_5pp_phased.png",
    "シナリオ3": "irf_plot_income_tax_reduction_5pp_permanent.png",
    "シナリオ4": "irf_plot_revenue_neutral_c_up_l_down.png",
}
image_tags_html = {}
for scenario_name_jp, plot_filename in plot_files.items():
    # Corrected path construction for HTML: should be relative to the HTML file's location if plots are in a subfolder
    # Assuming HTML file is in root, and plots are in results/current_analysis_report_data/
    plot_path_html_relative = os.path.join("results", "current_analysis_report_data", plot_filename)

    # Check existence using the actual filesystem path
    if os.path.exists(os.path.join(SIMULATION_DATA_DIR, plot_filename)):
        image_tags_html[scenario_name_jp] = f"<p><img src='{plot_path_html_relative}' alt='{scenario_name_jp} IRFプロット' style='width:100%;max-width:600px;border:1px solid #ccc;'></p>\n" # Added border
    else:
        image_tags_html[scenario_name_jp] = f"<p><i>{scenario_name_jp} のIRFプロット画像は見つかりませんでした。({plot_path_html_relative} を確認)</i></p>\n"

css_style_block = """<style>
body { font-family: 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; margin: 0; padding: 0; background-color: #f4f4f4; color: #333; }
.container { width: 80%; margin: auto; background-color: #fff; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
h1, h2, h3, h4 { color: #2c3e50; }
h1 { text-align: center; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
h2 { border-bottom: 1px solid #eee; padding-bottom: 5px; margin-top: 30px; }
h3 { margin-top: 25px; color: #3498db; }
h4 { margin-top: 20px; color: #555; }
table { border-collapse: collapse; width: auto; margin-top: 10px; margin-bottom: 15px; }
th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
th { background-color: #f9f9f9; }
ul { padding-left: 20px; }
li { margin-bottom: 5px; }
p { margin-bottom: 15px; }
.code { background-color: #e8e8e8; padding: 2px 5px; border-radius: 3px; font-family: 'Courier New', Courier, monospace; }
.note { background-color: #fff9c4; border-left: 4px solid #fdd835; padding: 10px; margin-top: 10px; }
img { border: 1px solid #ccc; padding: 3px; background-color: #fff; }
</style>"""
html_template = "<!DOCTYPE html>\n<html lang='ja'>\n<head>\n  <meta charset='UTF-8'>\n  <title>税制改革の経済効果分析レポート</title>\n{css_styles}\n</head>\n<body>\n  <div class='container'>\n{body_content}\n  </div>\n</body>\n</html>"

report_body_parts = []

try:
    with open(TEXT_REPORT_PATH, "r", encoding="utf-8") as f:
        text_report = f.read()
except FileNotFoundError:
    error_content = "<h1>エラー</h1><p>元のテキストレポートファイルが見つかりませんでした。</p>"
    if 'css_style_block' not in locals(): css_style_block = "" # Ensure defined
    html_content = html_template.format(css_styles=css_style_block, body_content=error_content)
    with open(HTML_OUTPUT_FILENAME, "w", encoding="utf-8") as f_html:
        f_html.write(html_content)
    print(f"Partial HTML report (error) written to {HTML_OUTPUT_FILENAME}")
    raise

sections = text_report.split("\n\n")
current_h2_title = ""

for section_text in sections:
    section_text = section_text.strip()
    if not section_text:
        continue

    if section_text.startswith("1. はじめに"):
        header_phrase = "1. はじめに"
        current_h2_title = header_phrase.split('. ', 1)[1] if '. ' in header_phrase else header_phrase
        content = section_text[len(header_phrase):].lstrip(":\n").strip()
        report_body_parts.append(f"<h1>税制改革の経済効果分析レポート</h1>\n<h2>{current_h2_title}</h2>\n<p>{content}</p>\n")
    elif section_text.startswith("2. モデル概要"):
        header_phrase = "2. モデル概要"
        current_h2_title = header_phrase.split('. ', 1)[1] if '. ' in header_phrase else header_phrase
        content = section_text[len(header_phrase):].lstrip(":\n").strip()
        replacement_html_config = '<span class="code">config/parameters.json</span>'
        content_config = content.replace('`config/parameters.json`', replacement_html_config)
        report_body_parts.append(f"<h2>{current_h2_title}</h2>\n<p>{content_config}</p>\n")
    elif section_text.startswith("3. シミュレーションシナリオ"):
        header_phrase = "3. シミュレーションシナリオ"
        current_h2_title = header_phrase.split('. ', 1)[1] if '. ' in header_phrase else header_phrase
        content_part = section_text[len(header_phrase):].lstrip(":\n").strip()
        report_body_parts.append(f"<h2>{current_h2_title}</h2>\n{text_to_html_list(content_part)}\n")
    elif section_text.startswith("4. シミュレーション結果"):
        current_h2_title = "シミュレーション結果" # This one is just a main header, no content extraction here
        report_body_parts.append(f"<h2>{current_h2_title}</h2>\n")
    elif section_text.startswith("【シナリオ1"):
        report_body_parts.append("<h3>シナリオ1: 恒久的消費税率5%ポイント引き上げ</h3>\n")
        report_body_parts.append(create_ss_table_html("consumption_tax_increase_5pp_permanent", "シナリオ1"))
        # Extracting IRF description and revenue note more robustly
        desc_parts = section_text.split("「恒久的消費税率5%ポイント引き上げ」の動学的調整過程の概要：", 1)
        if len(desc_parts) > 1:
            main_desc_part = desc_parts[1]
            revenue_note_parts = main_desc_part.split("特筆すべき点として、", 1)
            irf_text = revenue_note_parts[0].strip().replace("\n", "<br>\n")
            report_body_parts.append(f"<h4>動学的調整過程</h4>\n<p>{irf_text}</p>\n")
            if len(revenue_note_parts) > 1:
                 report_body_parts.append(f"<p class='note'>特筆すべき点として、{revenue_note_parts[1].strip()}</p>\n")
        report_body_parts.append(image_tags_html.get("シナリオ1", ""))
    elif section_text.startswith("【シナリオ2"):
        report_body_parts.append("<h3>シナリオ2: 段階的消費税率5%ポイント引き上げ</h3>\n")
        report_body_parts.append(create_ss_table_html("consumption_tax_increase_5pp_phased", "シナリオ2"))
        desc_parts = section_text.split("「段階的消費税率5%ポイント引き上げ」の動学的調整過程の概要：", 1)
        if len(desc_parts) > 1:
            main_desc_part = desc_parts[1]
            note_parts = main_desc_part.split("長期的（定常状態）な影響は", 1)
            irf_text = note_parts[0].strip().replace("\n", "<br>\n")
            report_body_parts.append(f"<h4>動学的調整過程</h4>\n<p>{irf_text}</p>\n")
            if len(note_parts) > 1:
                 report_body_parts.append(f"<p>長期的（定常状態）な影響は{note_parts[1].strip()}</p>\n")
        report_body_parts.append(image_tags_html.get("シナリオ2", ""))
    elif section_text.startswith("【シナリオ3"):
        report_body_parts.append("<h3>シナリオ3: 恒久的所得税率5%ポイント引き下げ</h3>\n")
        report_body_parts.append(create_ss_table_html("income_tax_reduction_5pp_permanent", "シナリオ3"))
        desc_parts = section_text.split("「恒久的所得税率5%ポイント引き下げ」の動学的調整過程の概要：", 1)
        if len(desc_parts) > 1:
            main_desc_part = desc_parts[1]
            revenue_note_parts = main_desc_part.split("所得税率引き下げにより経済活動が刺激され、", 1)
            irf_text = revenue_note_parts[0].strip().replace("\n", "<br>\n")
            report_body_parts.append(f"<h4>動学的調整過程</h4>\n<p>{irf_text}</p>\n")
            if len(revenue_note_parts) > 1:
                 report_body_parts.append(f"<p class='note'>所得税率引き下げにより経済活動が刺激され、{revenue_note_parts[1].strip()}</p>\n")
        report_body_parts.append(image_tags_html.get("シナリオ3", ""))
    elif section_text.startswith("【シナリオ4"):
        report_body_parts.append("<h3>シナリオ4: 税収中立的改革（消費税+2%p、所得税-2%p）</h3>\n")
        report_body_parts.append(create_ss_table_html("revenue_neutral_c_up_l_down", "シナリオ4"))
        desc_parts = section_text.split("「税収中立的改革」の動学的調整過程の概要：", 1)
        if len(desc_parts) > 1:
            main_desc_part = desc_parts[1]
            note_parts = main_desc_part.split("このシナリオでは、GDPが約", 1) # Problematic if this exact phrase isn't there
            irf_text = note_parts[0].strip().replace("\n", "<br>\n")
            report_body_parts.append(f"<h4>動学的調整過程</h4>\n<p>{irf_text}</p>\n")
            if len(note_parts) > 1: # This was the source of "このシナリオでは、GDPが約..."
                 report_body_parts.append(f"<p class='note'>このシナリオでは、GDPが約{note_parts[1].strip()}</p>\n")
            # The original text report for scenario 4 has more complex structure, this parsing is simplified
        report_body_parts.append(image_tags_html.get("シナリオ4", ""))

    elif section_text.startswith("5. 考察"):
        header_phrase = "5. 考察"
        current_h2_title = header_phrase.split('. ', 1)[1] if '. ' in header_phrase else header_phrase
        content = section_text[len(header_phrase):].lstrip(":\n").strip()
        content_paragraphs = content.split("\n") # Split by newline for paragraphs
        report_body_parts.append(f"<h2>{current_h2_title}</h2>\n")
        for para in content_paragraphs:
            if para.strip(): # Ensure paragraph is not just whitespace
                 replacement_html_arp = '<span class="code">ACADEMIC_RESEARCH_REMEDIATION_PLAN.md</span>'
                 content_arp = para.strip().replace('`ACADEMIC_RESEARCH_REMEDIATION_PLAN.md`', replacement_html_arp)
                 report_body_parts.append(f"<p>{content_arp}</p>\n")
    elif section_text.startswith("6. 結論"):
        header_phrase = "6. 結論"
        current_h2_title = header_phrase.split('. ', 1)[1] if '. ' in header_phrase else header_phrase
        content = section_text[len(header_phrase):].lstrip(":\n").strip()
        content_paragraphs = content.split("\n")
        report_body_parts.append(f"<h2>{current_h2_title}</h2>\n")
        for para in content_paragraphs:
            if para.strip():
                report_body_parts.append(f"<p>{para.strip()}</p>\n")
    elif section_text.startswith("7. 付録"):
        header_phrase = "7. 付録"
        current_h2_title = header_phrase.split('. ', 1)[1] if '. ' in header_phrase else header_phrase
        content = section_text[len(header_phrase):].lstrip(":\n").strip()
        replacement_html_results = '<span class="code">results/current_analysis_report_data/</span>'
        content_results = content.replace('`results/current_analysis_report_data/`', replacement_html_results)
        report_body_parts.append(f"<h2>{current_h2_title}</h2>\n<p>{content_results}</p>\n")
    # Fallback for content within section "4. シミュレーション結果" that is not a specific scenario
    elif current_h2_title == "シミュレーション結果" and not section_text.startswith("【シナリオ"):
         report_body_parts.append(f"<p>{section_text}</p>\n")


final_body_content = "".join(report_body_parts)
# The original script had a line here: html_output = html_output.replace("\n", "\n")
# This is redundant if html_output is already structured with newlines, or harmful if it removes intended newlines.
# Assuming final_body_content already has the correct HTML structure with necessary newlines.
html_output = html_template.format(css_styles=css_style_block, body_content=final_body_content)


with open(HTML_OUTPUT_FILENAME, "w", encoding="utf-8") as f:
    f.write(html_output)

print(f"HTML report content written to {HTML_OUTPUT_FILENAME}")
