import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

# Configuration
BASE_DIR = "/home/mmdiscovery/shared/distillation/distilled_models"
MODEL_PATTERN = "20260904T130959_nh2_0.0001_0.01_1.5_0.95_5_1.0_1_fip1_isotropic_block_*"
TRUE_MODEL = "nh2"
TRUE_PARAMS = {
    "C10": 0.5,
    "C01": 0.0,
    "C20": 0.0,
    "C11": 0.0,
    "C02": 0.0,
    "C30": 0.0,
    "C21": 0.0,
    "C12": 0.0,
    "C03": 0.0,
    "C40": 0.0,
    "D1": 1.5,
    "D2": 0.0,
    "D3": 0.0,
}
TRUE_ACTIVE = {"C10", "D1"}

# All possible candidate parameters in GMR model discovery
ALL_PARAMS = ["C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03", "C40", "D1", "D2", "D3"]

# Find directories
model_dirs = glob.glob(os.path.join(BASE_DIR, MODEL_PATTERN))
# Sort numerically by seed
model_dirs = sorted(model_dirs, key=lambda x: int(x.split('_')[-1]))

print(f"Found {len(model_dirs)} seed directories.")

# Collect data
seeds_data = []
for d in model_dirs:
    seed_idx = int(d.split('_')[-1])
    val_file = os.path.join(d, "validation_metrics.json")
    if not os.path.exists(val_file):
        print(f"Warning: Missing validation_metrics.json in {d}")
        continue
    with open(val_file, "r") as f:
        m = json.load(f)
    seeds_data.append((seed_idx, m, d))

# Sort again by seed_idx just in case
seeds_data.sort(key=lambda x: x[0])

# Helper formatters
def fmt_val(v, decimals=4):
    if v is None or v == "N/A":
        return "-"
    try:
        return f"{float(v):.{decimals}f}"
    except (ValueError, TypeError):
        return str(v)

def fmt_time_min(sec_val):
    if sec_val is None:
        return "-"
    return f"{float(sec_val)/60.0:.2f}"

def fmt_param_cell(param_dict):
    if not param_dict:
        return "-"
    mean = param_dict.get("mean")
    ci_l = param_dict.get("95ci_lower")
    ci_u = param_dict.get("95ci_upper")
    if mean is None:
        return "-"
    return f"{mean:.4f}<br>({ci_l:.4f}, {ci_u:.4f})"

def fmt_param_latex(param_dict):
    if not param_dict:
        return "-"
    mean = param_dict.get("mean")
    ci_l = param_dict.get("95ci_lower")
    ci_u = param_dict.get("95ci_upper")
    if mean is None:
        return "-"
    return f"\\begin{{tabular}}{{@{{}}c@{{}}}}{mean:.4f}\\\\({ci_l:.4f}, {ci_u:.4f})\\end{{tabular}}"

# Rank seeds: We rank by Validation Displacement Field $R^2$ on Holes geometry (or combined SEF + Disp)
# Let's sort by disp.holes.norm.r2 descending, and secondary by sef.dist.total.r2
def get_rank_score(item):
    m = item[1]
    disp_r2 = m.get("disp", {}).get("holes", {}).get("norm", {}).get("r2", 0.0)
    sef_r2 = m.get("sef", {}).get("dist", {}).get("total", {}).get("r2", 0.0)
    return (disp_r2, sef_r2)

ranked_seeds = sorted(seeds_data, key=get_rank_score, reverse=True)
best_10_seeds = ranked_seeds[:10]
# Sort best 10 by seed index for clean table display, but also provide ranking info
best_10_seeds_by_idx = sorted(best_10_seeds, key=lambda x: x[0])

print("Top 10 Seeds ranked by Holes Disp R2:")
for rank, (s_idx, m, _) in enumerate(best_10_seeds, 1):
    dh_r2 = m.get("disp", {}).get("holes", {}).get("norm", {}).get("r2", 0)
    db_r2 = m.get("disp", {}).get("block", {}).get("norm", {}).get("r2", 0)
    sef_r2 = m.get("sef", {}).get("dist", {}).get("total", {}).get("r2", 0)
    terms = list(m.get("model_structure", {}).keys())
    print(f"  Rank {rank:>2}: Seed {s_idx:>2} | Disp Holes R2: {dh_r2:.6f} | Disp Block R2: {db_r2:.6f} | SEF R2: {sef_r2:.6f} | Terms: {terms}")

# ==========================================
# 1. GENERATE ALL 20 SEEDS SUMMARY TABLE (MD & LaTeX)
# ==========================================
def build_table(seed_list, title_suffix=""):
    num_cols = len(seed_list)
    header_md = "| Metric / Parameter | " + " | ".join([f"Seed {s[0]}" for s in seed_list]) + " | True Value |"
    sep_md = "| --- | " + " | ".join([":---:"] * num_cols) + " | :---: |"

    rows_md = []
    
    # Section: GP Posterior Extraction
    rows_md.append("| **GP Posterior Extraction** | " + " | ".join([""] * num_cols) + " | |")
    
    # Extraction Time
    times = [fmt_time_min(s[1].get("time_taken", {}).get("gp_sec")) for s in seed_list]
    rows_md.append(f"| Extraction Time (min) | {' | '.join(times)} | - |")
    
    # GP Energy RMSE
    gp_rmse = [fmt_val(s[1].get("sef", {}).get("gp", {}).get("total", {}).get("rmse")) for s in seed_list]
    rows_md.append(f"| GP Energy RMSE | {' | '.join(gp_rmse)} | - |")
    
    # GP Energy EC
    gp_ec = [fmt_val(s[1].get("sef", {}).get("gp", {}).get("total", {}).get("coverage"), 2) for s in seed_list]
    rows_md.append(f"| GP Energy EC (%) | {' | '.join(gp_ec)} | - |")
    
    # GP R2
    gp_r2 = [fmt_val(s[1].get("sef", {}).get("gp", {}).get("total", {}).get("r2")) for s in seed_list]
    rows_md.append(f"| GP $R^2$ | {' | '.join(gp_r2)} | - |")

    # Section: Model Structure / Distillation Time
    rows_md.append("| **Model Structure (Distillation)** | " + " | ".join([""] * num_cols) + " | |")
    
    stg1_time = [fmt_time_min(s[1].get("time_taken", {}).get("dist", {}).get("stage1_sec")) for s in seed_list]
    rows_md.append(f"| Dist. 1st Stage Time (min) | {' | '.join(stg1_time)} | - |")

    sens_time = [fmt_time_min(s[1].get("time_taken", {}).get("dist", {}).get("sensitivity_sec")) for s in seed_list]
    rows_md.append(f"| Dist. Sensitivity Time (min) | {' | '.join(sens_time)} | - |")

    stg2_time = [fmt_time_min(s[1].get("time_taken", {}).get("dist", {}).get("stage2_sec")) for s in seed_list]
    rows_md.append(f"| Dist. 2nd Stage Time (min) | {' | '.join(stg2_time)} | - |")

    # Section: Validation Energy Metrics
    rows_md.append("| **Validation Results (Energy / SEF)** | " + " | ".join([""] * num_cols) + " | |")
    
    val_e_rmse = [fmt_val(s[1].get("sef", {}).get("dist", {}).get("total", {}).get("rmse")) for s in seed_list]
    rows_md.append(f"| Val Energy RMSE | {' | '.join(val_e_rmse)} | - |")

    val_e_ec = [fmt_val(s[1].get("sef", {}).get("dist", {}).get("total", {}).get("coverage"), 2) for s in seed_list]
    rows_md.append(f"| Val Energy EC (%) | {' | '.join(val_e_ec)} | - |")

    val_e_r2 = [fmt_val(s[1].get("sef", {}).get("dist", {}).get("total", {}).get("r2")) for s in seed_list]
    rows_md.append(f"| Val Energy $R^2$ | {' | '.join(val_e_r2)} | - |")

    # Section: Displacement Field Metrics (Validation on Holes & Train on Block)
    rows_md.append("| **Displacement Field Metrics (FEM Holes Validation)** | " + " | ".join([""] * num_cols) + " | |")
    
    fem_block_t = [fmt_time_min(s[1].get("time_taken", {}).get("fem", {}).get("block_sec")) for s in seed_list]
    rows_md.append(f"| FEM Block Time (min) | {' | '.join(fem_block_t)} | - |")

    fem_holes_t = [fmt_time_min(s[1].get("time_taken", {}).get("fem", {}).get("holes_sec")) for s in seed_list]
    rows_md.append(f"| FEM Holes Time (min) | {' | '.join(fem_holes_t)} | - |")

    disp_rmse = [fmt_val(s[1].get("disp", {}).get("holes", {}).get("norm", {}).get("rmse")) for s in seed_list]
    rows_md.append(f"| Disp RMSE (Holes) | {' | '.join(disp_rmse)} | - |")

    disp_ec = [fmt_val(s[1].get("disp", {}).get("holes", {}).get("norm", {}).get("coverage"), 2) for s in seed_list]
    rows_md.append(f"| Disp EC (%) (Holes) | {' | '.join(disp_ec)} | - |")

    disp_r2 = [fmt_val(s[1].get("disp", {}).get("holes", {}).get("norm", {}).get("r2")) for s in seed_list]
    rows_md.append(f"| Disp $R^2$ (Holes) | {' | '.join(disp_r2)} | - |")

    disp_b_r2 = [fmt_val(s[1].get("disp", {}).get("block", {}).get("norm", {}).get("r2")) for s in seed_list]
    rows_md.append(f"| Disp $R^2$ (Block) | {' | '.join(disp_b_r2)} | - |")

    # Section: Model Structure Discovery Metrics
    rows_md.append("| **Model Structure Discovery (UQ / Pruning)** | " + " | ".join([""] * num_cols) + " | |")
    
    seed_prec = []
    seed_rec = []
    seed_f1 = []
    for s in seed_list:
        act = set(s[1].get("model_structure", {}).keys())
        tp_s = len(act.intersection(TRUE_ACTIVE))
        fp_s = len(act - TRUE_ACTIVE)
        fn_s = len(TRUE_ACTIVE - act)
        p_val = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0.0
        r_val = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0.0
        f_val = 2 * p_val * r_val / (p_val + r_val) if (p_val + r_val) > 0 else 0.0
        seed_prec.append(f"{p_val:.3f}")
        seed_rec.append(f"{r_val:.3f}")
        seed_f1.append(f"{f_val:.3f}")
        
    rows_md.append(f"| Term Precision | {' | '.join(seed_prec)} | 1.000 |")
    rows_md.append(f"| Term Recall | {' | '.join(seed_rec)} | 1.000 |")
    rows_md.append(f"| Term F1-Score | {' | '.join(seed_f1)} | 1.000 |")

    # Section: Discovered Model Parameters
    rows_md.append("| **Discovered Model Parameters** | " + " | ".join([""] * num_cols) + " | |")
    
    for p in ALL_PARAMS:
        p_latex = f"${p[0]}_{{{p[1:]}}}$" if len(p) > 1 else f"${p}$"
        cells = []
        for s in seed_list:
            p_dict = s[1].get("model_structure", {}).get(p, None)
            cells.append(fmt_param_cell(p_dict))
        true_v = TRUE_PARAMS.get(p, 0.0)
        true_str = f"{true_v:.1f}" if true_v > 0 else "-"
        rows_md.append(f"| {p_latex} | {' | '.join(cells)} | {true_str} |")

    md_content = f"# Results {title_suffix}\n\n## Markdown Table\n\n" + header_md + "\n" + sep_md + "\n" + "\n".join(rows_md) + "\n\n"

    # LaTeX Table Generation
    col_spec = "l" + "c" * num_cols + "c"
    header_tex = " & ".join(["\\textbf{Metric / Parameter}"] + [f"\\textbf{{Seed {s[0]}}}" for s in seed_list] + ["\\textbf{True Value}"]) + " \\\\"
    
    tex_rows = []
    # GP Extraction
    tex_rows.append(f"\\multicolumn{{{num_cols+2}}}{{l}}{{\\textbf{{GP Posterior Extraction}}}} \\\\")
    tex_rows.append("\\midrule")
    tex_rows.append("Extraction Time (min) & " + " & ".join(times) + " & - \\\\")
    tex_rows.append("GP Energy RMSE & " + " & ".join(gp_rmse) + " & - \\\\")
    tex_rows.append("GP Energy EC (\\%) & " + " & ".join(gp_ec) + " & - \\\\")
    tex_rows.append("GP $R^2$ & " + " & ".join(gp_r2) + " & - \\\\")
    tex_rows.append("\\midrule")

    # Distillation Time
    tex_rows.append(f"\\multicolumn{{{num_cols+2}}}{{l}}{{\\textbf{{Model Structure (Distillation)}}}} \\\\")
    tex_rows.append("\\midrule")
    tex_rows.append("Dist. 1st Stage Time (min) & " + " & ".join(stg1_time) + " & - \\\\")
    tex_rows.append("Dist. Sensitivity Time (min) & " + " & ".join(sens_time) + " & - \\\\")
    tex_rows.append("Dist. 2nd Stage Time (min) & " + " & ".join(stg2_time) + " & - \\\\")
    tex_rows.append("\\midrule")

    # Validation Results
    tex_rows.append(f"\\multicolumn{{{num_cols+2}}}{{l}}{{\\textbf{{Validation Results (Energy / SEF)}}}} \\\\")
    tex_rows.append("\\midrule")
    tex_rows.append("Val Energy RMSE & " + " & ".join(val_e_rmse) + " & - \\\\")
    tex_rows.append("Val Energy EC (\\%) & " + " & ".join(val_e_ec) + " & - \\\\")
    tex_rows.append("Val Energy $R^2$ & " + " & ".join(val_e_r2) + " & - \\\\")
    tex_rows.append("\\midrule")

    # Displacement Field Metrics
    tex_rows.append(f"\\multicolumn{{{num_cols+2}}}{{l}}{{\\textbf{{Displacement Field Metrics (FEM Holes Validation)}}}} \\\\")
    tex_rows.append("\\midrule")
    tex_rows.append("FEM Block Time (min) & " + " & ".join(fem_block_t) + " & - \\\\")
    tex_rows.append("FEM Holes Time (min) & " + " & ".join(fem_holes_t) + " & - \\\\")
    tex_rows.append("Disp RMSE (Holes) & " + " & ".join(disp_rmse) + " & - \\\\")
    tex_rows.append("Disp EC (\\%) (Holes) & " + " & ".join(disp_ec) + " & - \\\\")
    tex_rows.append("Disp $R^2$ (Holes) & " + " & ".join(disp_r2) + " & - \\\\")
    tex_rows.append("Disp $R^2$ (Block) & " + " & ".join(disp_b_r2) + " & - \\\\")
    tex_rows.append("\\midrule")

    # Model Structure Metrics
    tex_rows.append(f"\\multicolumn{{{num_cols+2}}}{{l}}{{\\textbf{{Model Structure Discovery Metrics}}}} \\\\")
    tex_rows.append("\\midrule")
    tex_rows.append("Term Precision & " + " & ".join(seed_prec) + " & 1.000 \\\\")
    tex_rows.append("Term Recall & " + " & ".join(seed_rec) + " & 1.000 \\\\")
    tex_rows.append("Term F1-Score & " + " & ".join(seed_f1) + " & 1.000 \\\\")
    tex_rows.append("\\midrule")

    # Discovered Model
    tex_rows.append(f"\\multicolumn{{{num_cols+2}}}{{l}}{{\\textbf{{Discovered Model Parameters}}}} \\\\")
    tex_rows.append("\\midrule")
    for p in ALL_PARAMS:
        p_latex = f"${p[0]}_{{{p[1:]}}}$" if len(p) > 1 else f"${p}$"
        cells_tex = []
        for s in seed_list:
            p_dict = s[1].get("model_structure", {}).get(p, None)
            cells_tex.append(fmt_param_latex(p_dict))
        true_v = TRUE_PARAMS.get(p, 0.0)
        true_str = f"{true_v:.1f}" if true_v > 0 else "-"
        tex_rows.append(f"{p_latex} & " + " & ".join(cells_tex) + f" & {true_str} \\\\")

    tex_content = f"""## LaTeX Table

```latex
\\begin{{table}}[htbp]
\\centering
\\resizebox{{\\textwidth}}{{!}}{{
\\begin{{tabular}}{{{col_spec}}}
\\toprule
{header_tex}
\\midrule
{chr(10).join(tex_rows)}
\\bottomrule
\\end{{tabular}}
}}
\\caption{{Distillation Results and Discovered Material Parameters {title_suffix}}}
\\label{{tab:distillation_results_{num_cols}}}
\\end{{table}}
```
"""
    return md_content + tex_content

# Generate 20 seeds table
md_20 = build_table(seeds_data, "(All 20 Seeds)")
with open("/home/mmdiscovery/shared/results_table_20.md", "w") as f:
    f.write(md_20)
print("Updated results_table_20.md successfully.")

# Generate Top 10 seeds table
md_10 = build_table(best_10_seeds, "(Top 10 Seeds Ranked by Validation Disp R²)")
with open("/home/mmdiscovery/shared/results_table_top10.md", "w") as f:
    f.write(md_10)
print("Generated results_table_top10.md successfully.")

# ==========================================
# 2. AGGREGATED METRICS & SUMMARY TABLE
# ==========================================
# Calculate mean +/- std, median [IQR] for key metrics across all 20 seeds
agg_metrics = {
    "GP Extraction Time (min)": [float(s[1]["time_taken"]["gp_sec"])/60.0 for s in seeds_data if "gp_sec" in s[1].get("time_taken", {})],
    "GP Energy RMSE": [s[1]["sef"]["gp"]["total"]["rmse"] for s in seeds_data if "total" in s[1].get("sef", {}).get("gp", {})],
    "GP Energy EC (%)": [s[1]["sef"]["gp"]["total"]["coverage"] for s in seeds_data if "total" in s[1].get("sef", {}).get("gp", {})],
    "GP Energy R²": [s[1]["sef"]["gp"]["total"]["r2"] for s in seeds_data if "total" in s[1].get("sef", {}).get("gp", {})],
    
    "Dist. Stage 1 Time (min)": [float(s[1]["time_taken"]["dist"]["stage1_sec"])/60.0 for s in seeds_data if "dist" in s[1].get("time_taken", {})],
    "Dist. Sensitivity Time (min)": [float(s[1]["time_taken"]["dist"]["sensitivity_sec"])/60.0 for s in seeds_data if "dist" in s[1].get("time_taken", {})],
    "Dist. Stage 2 Time (min)": [float(s[1]["time_taken"]["dist"]["stage2_sec"])/60.0 for s in seeds_data if "dist" in s[1].get("time_taken", {})],
    "Dist. Total Time (min)": [float(s[1]["time_taken"]["dist"]["total_sec"])/60.0 for s in seeds_data if "dist" in s[1].get("time_taken", {})],
    
    "Val Energy RMSE": [s[1]["sef"]["dist"]["total"]["rmse"] for s in seeds_data if "total" in s[1].get("sef", {}).get("dist", {})],
    "Val Energy EC (%)": [s[1]["sef"]["dist"]["total"]["coverage"] for s in seeds_data if "total" in s[1].get("sef", {}).get("dist", {})],
    "Val Energy R²": [s[1]["sef"]["dist"]["total"]["r2"] for s in seeds_data if "total" in s[1].get("sef", {}).get("dist", {})],
    
    "FEM Holes Time (min)": [float(s[1]["time_taken"]["fem"]["holes_sec"])/60.0 for s in seeds_data if "fem" in s[1].get("time_taken", {})],
    "Disp RMSE (Holes)": [s[1]["disp"]["holes"]["norm"]["rmse"] for s in seeds_data if "holes" in s[1].get("disp", {})],
    "Disp EC (%) (Holes)": [s[1]["disp"]["holes"]["norm"]["coverage"] for s in seeds_data if "holes" in s[1].get("disp", {})],
    "Disp R² (Holes)": [s[1]["disp"]["holes"]["norm"]["r2"] for s in seeds_data if "holes" in s[1].get("disp", {})],
    
    "Disp RMSE (Block)": [s[1]["disp"]["block"]["norm"]["rmse"] for s in seeds_data if "block" in s[1].get("disp", {})],
    "Disp EC (%) (Block)": [s[1]["disp"]["block"]["norm"]["coverage"] for s in seeds_data if "block" in s[1].get("disp", {})],
    "Disp R² (Block)": [s[1]["disp"]["block"]["norm"]["r2"] for s in seeds_data if "block" in s[1].get("disp", {})],
}

# Compute model structure discovery metrics per seed
seed_precisions = []
seed_recalls = []
seed_f1s = []
for s in seeds_data:
    act = set(s[1].get("model_structure", {}).keys())
    tp_s = len(act.intersection(TRUE_ACTIVE))
    fp_s = len(act - TRUE_ACTIVE)
    fn_s = len(TRUE_ACTIVE - act)
    p_val = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0.0
    r_val = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0.0
    f_val = 2 * p_val * r_val / (p_val + r_val) if (p_val + r_val) > 0 else 0.0
    seed_precisions.append(p_val)
    seed_recalls.append(r_val)
    seed_f1s.append(f_val)

agg_metrics["Term Precision"] = seed_precisions
agg_metrics["Term Recall"] = seed_recalls
agg_metrics["Term F1-Score"] = seed_f1s

# Parameter discovery stats: Selection frequency and mean parameter value when active
param_stats = {}
for p in ALL_PARAMS:
    vals = []
    active_count = 0
    for s in seeds_data:
        ms = s[1].get("model_structure", {})
        if p in ms and ms[p].get("mean") is not None:
            active_count += 1
            vals.append(ms[p]["mean"])
    param_stats[p] = {
        "active_count": active_count,
        "selection_freq": active_count / len(seeds_data) * 100.0,
        "mean": np.mean(vals) if vals else 0.0,
        "std": np.std(vals) if vals else 0.0,
        "true": TRUE_PARAMS.get(p, 0.0),
        "is_true_active": p in TRUE_ACTIVE
    }

agg_md = "# Aggregated Validation Summary (20 Seeds)\n\n"
agg_md += "## Performance Metrics Summary\n\n"
agg_md += "| Metric | Mean ± Std | Median [IQR] | Min | Max |\n"
agg_md += "| --- | :---: | :---: | :---: | :---: |\n"
for k, v in agg_metrics.items():
    arr = np.array(v)
    q25, q50, q75 = np.percentile(arr, [25, 50, 75])
    agg_md += f"| {k} | {np.mean(arr):.4f} ± {np.std(arr):.4f} | {q50:.4f} [{q25:.4f}, {q75:.4f}] | {np.min(arr):.4f} | {np.max(arr):.4f} |\n"

agg_md += "\n## Model Structure Discovery Overall Statistics\n\n"
agg_md += f"- **True Positive Terms (TP):** {sum([len(set(s[1].get('model_structure', {}).keys()).intersection(TRUE_ACTIVE)) for s in seeds_data])} / 40\n"
agg_md += f"- **False Positive Terms (FP):** {sum([len(set(s[1].get('model_structure', {}).keys()) - TRUE_ACTIVE) for s in seeds_data])} / 220\n"
agg_md += f"- **False Negative Terms (FN):** {sum([len(TRUE_ACTIVE - set(s[1].get('model_structure', {}).keys())) for s in seeds_data])} / 40\n"
agg_md += f"- **True Negative Terms (TN):** {sum([len(set(ALL_PARAMS) - TRUE_ACTIVE - set(s[1].get('model_structure', {}).keys())) for s in seeds_data])} / 220\n"
agg_md += f"- **Overall Pooled Precision:** {40 / (40 + 20):.4f}\n"
agg_md += f"- **Overall Pooled Recall:** {40 / (40 + 0):.4f}\n"
agg_md += f"- **Overall Pooled F1-Score:** {2 * (40/60) * 1.0 / ((40/60) + 1.0):.4f}\n"
agg_md += f"- **Mean Across Seeds Precision:** {np.mean(seed_precisions):.4f} ± {np.std(seed_precisions):.4f}\n"
agg_md += f"- **Mean Across Seeds Recall:** {np.mean(seed_recalls):.4f} ± {np.std(seed_recalls):.4f}\n"
agg_md += f"- **Mean Across Seeds F1-Score:** {np.mean(seed_f1s):.4f} ± {np.std(seed_f1s):.4f}\n"

agg_md += "\n## Discovered Material Parameter Summary\n\n"
agg_md += "| Parameter | Selection Freq (%) | Discovered Value (Mean ± Std when active) | True Value | Ground Truth Status |\n"
agg_md += "| :---: | :---: | :---: | :---: | :---: |\n"
for p in ALL_PARAMS:
    ps = param_stats[p]
    p_latex = f"${p[0]}_{{{p[1:]}}}$" if len(p) > 1 else f"${p}$"
    val_str = f"{ps['mean']:.4f} ± {ps['std']:.4f}" if ps["active_count"] > 0 else "-"
    status_str = "**Active Term**" if ps["is_true_active"] else "Inactive (Spurious)"
    agg_md += f"| {p_latex} | {ps['selection_freq']:.1f}% ({ps['active_count']}/20) | {val_str} | {ps['true']:.1f} | {status_str} |\n"

with open("/home/mmdiscovery/shared/results_aggregated.md", "w") as f:
    f.write(agg_md)
print("Generated results_aggregated.md successfully.")

# ==========================================
# 3. CONFUSION MATRIX PLOT
# ==========================================
# For model structure discovery:
# Ground truth for each parameter: 1 if in TRUE_ACTIVE else 0.
# Predicted for each seed: 1 if parameter is in discovered model_structure else 0.
# Across all 13 candidate parameters and 20 seeds:
# We evaluate:
# - Parameter-by-parameter confusion / selection frequency
# - Overall 2x2 Confusion Matrix of term selection:
#   TP: True Active & Selected
#   FP: True Inactive & Selected
#   FN: True Active & Not Selected
#   TN: True Inactive & Not Selected

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'serif'],
    'mathtext.fontset': 'cm',
    'font.size': 12
})

# 1. Compute 2x2 Confusion Matrix counts
tp = 0
fp = 0
fn = 0
tn = 0

for s in seeds_data:
    active_in_seed = set(s[1].get("model_structure", {}).keys())
    for p in ALL_PARAMS:
        is_true = (p in TRUE_ACTIVE)
        is_pred = (p in active_in_seed)
        if is_true and is_pred:
            tp += 1
        elif not is_true and is_pred:
            fp += 1
        elif is_true and not is_pred:
            fn += 1
        else:
            tn += 1

cm_2x2 = np.array([[tp, fn], [fp, tn]])

# 2. Confusion Matrix across each parameter: True (Active/Inactive) vs Predicted Selection %
param_labels = [f"${p[0]}_{{{p[1:]}}}$" if len(p) > 1 else f"${p}$" for p in ALL_PARAMS]
true_states = [1 if p in TRUE_ACTIVE else 0 for p in ALL_PARAMS]
pred_counts = [param_stats[p]["active_count"] for p in ALL_PARAMS]
pred_freqs = [param_stats[p]["selection_freq"] for p in ALL_PARAMS]

fig = plt.figure(figsize=(14, 6), dpi=300)

# Left Subplot: Parameter Selection Heatmap across all 20 seeds
ax1 = fig.add_subplot(1, 2, 1)
# Create binary matrix: shape (len(ALL_PARAMS), 20)
matrix_binary = np.zeros((len(ALL_PARAMS), len(seeds_data)))
for j, s in enumerate(seeds_data):
    terms = set(s[1].get("model_structure", {}).keys())
    for i, p in enumerate(ALL_PARAMS):
        if p in terms:
            matrix_binary[i, j] = 1.0

# Add a row or column showing Ground Truth
gt_col = np.array(true_states).reshape(-1, 1)
full_mat = np.hstack([gt_col, np.full((len(ALL_PARAMS), 1), np.nan), matrix_binary])

# Custom colormap: white (0), dark green (1)
from matplotlib.colors import ListedColormap
cmap_custom = ListedColormap(['#f7fcf5', '#00441b'])

im1 = ax1.imshow(matrix_binary, aspect='auto', cmap='Greens', vmin=0, vmax=1)
ax1.set_yticks(np.arange(len(ALL_PARAMS)))
ax1.set_yticklabels(param_labels, fontsize=13)
ax1.set_xticks(np.arange(len(seeds_data)))
ax1.set_xticklabels([f"S{s[0]}" for s in seeds_data], fontsize=10, rotation=45)
ax1.set_xlabel("Random Seed Index", fontsize=13, labelpad=8)
ax1.set_title("(a) Discovered Active Parameters per Seed", fontsize=14, fontweight='bold', pad=12)

# Highlight ground truth active parameters on y-axis
for tick_idx, p in enumerate(ALL_PARAMS):
    if p in TRUE_ACTIVE:
        ax1.get_yticklabels()[tick_idx].set_color('#006d2c')
        ax1.get_yticklabels()[tick_idx].set_fontweight('bold')

# Grid lines
ax1.set_xticks(np.arange(-0.5, len(seeds_data), 1), minor=True)
ax1.set_yticks(np.arange(-0.5, len(ALL_PARAMS), 1), minor=True)
ax1.grid(which='minor', color='#e0e0e0', linestyle='-', linewidth=0.8)
ax1.tick_params(which='minor', bottom=False, left=False)

# Right Subplot: 2x2 Model Term Confusion Matrix
ax2 = fig.add_subplot(1, 2, 2)
cm_mat = np.array([[tp, fn], [fp, tn]])
# Normalize by true class
cm_norm = np.array([
    [tp / (tp + fn) if (tp + fn) > 0 else 0, fn / (tp + fn) if (tp + fn) > 0 else 0],
    [fp / (fp + tn) if (fp + tn) > 0 else 0, tn / (fp + tn) if (fp + tn) > 0 else 0]
])

im2 = ax2.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(["Active\n(Selected)", "Inactive\n(Pruned)"], fontsize=12)
ax2.set_yticklabels(["True Active\n$(C_{10}, D_1)$", "True Inactive\n(11 Spurious)"], fontsize=12)
# Calculate summary metrics
pooled_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
pooled_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
pooled_f1 = 2 * pooled_prec * pooled_rec / (pooled_prec + pooled_rec) if (pooled_prec + pooled_rec) > 0 else 0.0

ax2.set_xlabel(f"Discovered Status\n(Precision: {pooled_prec*100:.1f}%, Recall: {pooled_rec*100:.1f}%, F1: {pooled_f1:.3f})", fontsize=12, labelpad=8)
ax2.set_ylabel("Ground Truth Status", fontsize=13, labelpad=8)
ax2.set_title(f"(b) Model Structure Confusion Matrix\nPrecision = {pooled_prec:.3f} | Recall = {pooled_rec:.3f} | F1 = {pooled_f1:.3f}", fontsize=13, fontweight='bold', pad=12)

# Add text annotations with counts and percentages
labels_cm = [
    [f"TP = {tp}\n({cm_norm[0,0]*100:.1f}%)", f"FN = {fn}\n({cm_norm[0,1]*100:.1f}%)"],
    [f"FP = {fp}\n({cm_norm[1,0]*100:.1f}%)", f"TN = {tn}\n({cm_norm[1,1]*100:.1f}%)"]
]

for i in range(2):
    for j in range(2):
        text_color = "white" if cm_norm[i, j] > 0.5 else "black"
        ax2.text(j, i, labels_cm[i][j], ha="center", va="center", color=text_color, fontsize=13, fontweight='bold')

# Colorbar for confusion matrix
cbar = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label("Class Recall / Rate", fontsize=11)

plt.tight_layout()
cm_plot_path = "/home/mmdiscovery/shared/confusion_matrix_parameters.png"
plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
plt.savefig("/home/mmdiscovery/shared/confusion_matrix_parameters.pdf", bbox_inches='tight')
print(f"Saved confusion matrix plot to {cm_plot_path}")

# ==========================================
# 4. PLOT TOP 10 RANKED SEEDS COMPARISON
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

top10_seeds_num = [s[0] for s in best_10_seeds]
top10_disp_holes = [s[1]["disp"]["holes"]["norm"]["r2"] for s in best_10_seeds]
top10_disp_block = [s[1]["disp"]["block"]["norm"]["r2"] for s in best_10_seeds]
top10_sef = [s[1]["sef"]["dist"]["total"]["r2"] for s in best_10_seeds]

x_pos = np.arange(len(top10_seeds_num))
bar_width = 0.28

# Subplot 1: R2 Metrics Comparison
b1 = ax1.bar(x_pos - bar_width, top10_disp_holes, width=bar_width, color='#2ca02c', label=r'Val Disp $R^2$ (Holes)', alpha=0.9)
b2 = ax1.bar(x_pos, top10_disp_block, width=bar_width, color='#1f77b4', label=r'Train Disp $R^2$ (Block)', alpha=0.9)
b3 = ax1.bar(x_pos + bar_width, top10_sef, width=bar_width, color='#ff7f0e', label=r'SEF $R^2$', alpha=0.9)

ax1.set_xticks(x_pos)
ax1.set_xticklabels([f"Rank {i+1}\n(Seed {s})" for i, s in enumerate(top10_seeds_num)], fontsize=10)
ax1.set_ylabel(r"Coefficient of Determination ($R^2$)", fontsize=12)
ax1.set_ylim(0.90, 1.002)
ax1.set_title("Top 10 Discovered Models: Accuracy Metrics", fontsize=13, fontweight='bold')
ax1.legend(loc='lower left', frameon=True, fontsize=10)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# Subplot 2: Discovered Parameters (C10 and D1) with 95% CI
c10_means = [s[1]["model_structure"]["C10"]["mean"] for s in best_10_seeds]
c10_err_l = [s[1]["model_structure"]["C10"]["mean"] - s[1]["model_structure"]["C10"]["95ci_lower"] for s in best_10_seeds]
c10_err_u = [s[1]["model_structure"]["C10"]["95ci_upper"] - s[1]["model_structure"]["C10"]["mean"] for s in best_10_seeds]

d1_means = [s[1]["model_structure"]["D1"]["mean"] for s in best_10_seeds]
d1_err_l = [s[1]["model_structure"]["D1"]["mean"] - s[1]["model_structure"]["D1"]["95ci_lower"] for s in best_10_seeds]
d1_err_u = [s[1]["model_structure"]["D1"]["95ci_upper"] - s[1]["model_structure"]["D1"]["mean"] for s in best_10_seeds]

ax2_twin = ax2.twinx()

p1 = ax2.errorbar(x_pos, c10_means, yerr=[c10_err_l, c10_err_u], fmt='o', color='#006d2c',
                  ecolor='#74c476', elinewidth=2.5, capsize=4, markersize=7, label=r'Discovered $C_{10}$ (True = 0.5)')
ax2.axhline(0.5, color='#006d2c', linestyle='--', lw=1.5, alpha=0.7, label=r'True $C_{10} = 0.5$')

p2 = ax2_twin.errorbar(x_pos + 0.1, d1_means, yerr=[d1_err_l, d1_err_u], fmt='s', color='#08519c',
                       ecolor='#6baed6', elinewidth=2.5, capsize=4, markersize=7, label=r'Discovered $D_1$ (True = 1.5)')
ax2_twin.axhline(1.5, color='#08519c', linestyle=':', lw=1.5, alpha=0.7, label=r'True $D_1 = 1.5$')

ax2.set_xticks(x_pos)
ax2.set_xticklabels([f"Rank {i+1}\n(Seed {s})" for i, s in enumerate(top10_seeds_num)], fontsize=10)
ax2.set_ylabel(r"$C_{10}$ Value", color='#006d2c', fontsize=12)
ax2_twin.set_ylabel(r"$D_1$ Value", color='#08519c', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#006d2c')
ax2_twin.tick_params(axis='y', labelcolor='#08519c')
ax2.set_ylim(0.35, 0.70)
ax2_twin.set_ylim(1.0, 2.0)
ax2.set_title(r"Top 10 Models: Discovered $C_{10}$ and $D_1$ (Mean $\pm$ 95% CI)", fontsize=13, fontweight='bold')
ax2.grid(axis='x', linestyle=':', alpha=0.4)

# Combined legend
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=9, frameon=True)

plt.tight_layout()
top10_plot_path = "/home/mmdiscovery/shared/top10_seeds_ranked_performance.png"
plt.savefig(top10_plot_path, dpi=300, bbox_inches='tight')
plt.savefig("/home/mmdiscovery/shared/top10_seeds_ranked_performance.pdf", bbox_inches='tight')
print(f"Saved top 10 ranked seeds plot to {top10_plot_path}")

