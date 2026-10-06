#!/usr/bin/env python3
"""
Generate a Cross-Model Aggregate Summary Table (Markdown + LaTeX)
comparing multiple constitutive discovery experiments (e.g. Neo-Hookean, Gent-Thomas, Isihara).

For each model:
1. Aggregates all quantitative metrics across all seeds as Mean ± Std.
2. Reports discovered candidate material parameters from the median model (with 95% CI)
   alongside the ground-truth parameter values.

Usage:
    python3 plots/generate_cross_model_summary.py \
        --exp_dirs /path/to/nh2 /path/to/gentthomas \
        --labels nh2 gentthomas \
        --out_md results/cross_model_summary.md
"""

import os
import json
import yaml
import argparse
import numpy as np

ALL_CANDIDATE_PARAMS = ["C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03", "E", "D1", "D2", "D3"]
DEV_PARAMS_SET = {"C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03", "E"}
VOL_PARAMS_SET = {"D1", "D2", "D3"}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Cross-Model Aggregate Summary Table")
    parser.add_argument("--exp_dirs", nargs="+", required=True, help="List of experiment directory paths")
    parser.add_argument("--labels", nargs="+", default=None, help="Optional short labels for each experiment")
    parser.add_argument("--out_md", type=str, default="/home/mmdiscovery/shared/results/cross_model_summary.md",
                        help="Path to output markdown file")
    return parser.parse_args()


def load_experiment_config(exp_dir):
    cfg_candidates = [
        os.path.join(exp_dir, "config.yaml"),
        os.path.join(exp_dir, "config.json"),
    ]
    for c in cfg_candidates:
        if os.path.exists(c):
            with open(c, "r") as f:
                if c.endswith(".yaml") or c.endswith(".yml"):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
    return {}


def find_seed_dirs(exp_dir):
    seeds = []
    if not os.path.isdir(exp_dir):
        return seeds
    for entry in os.listdir(exp_dir):
        full_path = os.path.join(exp_dir, entry)
        if os.path.isdir(full_path) and entry.isdigit():
            seeds.append((int(entry), full_path))
    seeds.sort(key=lambda x: x[0])
    return seeds


def load_seed_metrics(seed_path):
    candidates = [
        os.path.join(seed_path, "distilled", "validation_metrics.json"),
        os.path.join(seed_path, "distilled", "validation_metrics_gmr.json"),
        os.path.join(seed_path, "validation_metrics.json"),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            with open(cand, "r") as f:
                data = json.load(f)
            found_path = cand
            # Merge reaction force metrics if available
            for geom in ["block", "holes", "ttc"]:
                for cand_rf in [
                    os.path.join(seed_path, "fem_validation", geom, f"reaction_force_metrics_{geom}.json"),
                    os.path.join(seed_path, "distilled", f"reaction_force_metrics_{geom}.json"),
                    os.path.join(seed_path, "fem_validation", f"reaction_force_metrics_{geom}.json"),
                ]:
                    if os.path.exists(cand_rf):
                        try:
                            with open(cand_rf, "r") as rf_file:
                                rf_m = json.load(rf_file)
                            if "force" not in data:
                                data["force"] = {}
                            data["force"][geom] = rf_m
                            break
                        except Exception:
                            pass
            return data, found_path
    return None, None


def rank_seeds(seed_data_list):
    def score_seed(item):
        m = item.get("metrics", {})
        disp_dict = m.get("disp", {}) if isinstance(m, dict) else {}
        disp_b = (disp_dict.get("block") or disp_dict.get("ttc") or disp_dict.get("holes") or {}) if isinstance(disp_dict, dict) else {}
        norm_b = disp_b.get("norm", {}) if isinstance(disp_b, dict) else {}
        r2 = norm_b.get("r2", -999.0)
        rmse = norm_b.get("rmse", 999.0)
        cov = disp_b.get("coverage_xy", norm_b.get("coverage", 0.0))
        return (r2, -rmse, -abs(cov - 95.0))

    sorted_list = sorted(seed_data_list, key=score_seed, reverse=True)
    for rank, item in enumerate(sorted_list, start=1):
        item["rank"] = rank
    return sorted_list


def get_val(d, *keys):
    curr = d.get("metrics", {})
    for k in keys:
        if not isinstance(curr, dict) or k not in curr:
            return None
        curr = curr[k]
    return curr


def get_agg_stat(seeds, *keys, decimals=4, is_pct=False):
    vals = []
    for s in seeds:
        v = get_val(s, *keys)
        if v is not None and v != "N/A":
            try:
                fv = float(v)
                if not np.isnan(fv):
                    vals.append(fv)
            except (ValueError, TypeError):
                pass
    if not vals:
        statuses = [get_val(s, *keys) for s in seeds]
        valid_statuses = [st for st in statuses if st is not None]
        if valid_statuses:
            zero_cnt = sum(1 for st in valid_statuses if "Zero" in str(st))
            return f"{zero_cnt}/{len(valid_statuses)} Zero Discrepancy"
        return "-"

    mean_v = np.mean(vals)
    std_v = np.std(vals)
    if is_pct:
        return f"{mean_v:.2f} ± {std_v:.2f}"
    if 0 < abs(mean_v) < 1e-3:
        return f"{mean_v:.2e} ± {std_v:.2e}"
    return f"{mean_v:.{decimals}f} ± {std_v:.{decimals}f}"


def get_agg_stat_tex(seeds, *keys, decimals=4, is_pct=False):
    res = get_agg_stat(seeds, *keys, decimals=decimals, is_pct=is_pct)
    if res == "-":
        return "-"
    return res.replace("±", r"\pm")


def get_true_model_info(config):
    mat_model = config.get("material_model_name", "nh2").lower()
    true_params = {p: "-" for p in ALL_CANDIDATE_PARAMS}
    true_active = set()

    mat_p = config.get("material_params", {})
    dev_p = mat_p.get("dev_params", config.get("dev_params", None))
    vol_p = mat_p.get("vol_params", config.get("vol_params", None))

    if mat_model in ["nh2", "neohookean2", "neohookean"]:
        true_params["C10"] = "0.5" if not (dev_p and len(dev_p) > 0) else f"{float(dev_p[0]):.4f}".rstrip('0').rstrip('.')
        true_params["D1"] = "1.5" if not (vol_p and len(vol_p) > 0) else f"{float(vol_p[0]):.4f}".rstrip('0').rstrip('.')
        true_active = {"C10", "D1"}
    elif mat_model == "isihara":
        true_params["C10"] = "0.5" if not (dev_p and len(dev_p) > 0) else f"{float(dev_p[0]):.4f}".rstrip('0').rstrip('.')
        true_params["C01"] = "0.1" if not (dev_p and len(dev_p) > 1) else f"{float(dev_p[1]):.4f}".rstrip('0').rstrip('.')
        true_params["C20"] = "0.05" if not (dev_p and len(dev_p) > 2) else f"{float(dev_p[2]):.4f}".rstrip('0').rstrip('.')
        true_params["D1"] = "1.5" if not (vol_p and len(vol_p) > 0) else f"{float(vol_p[0]):.4f}".rstrip('0').rstrip('.')
        true_active = {"C10", "C01", "C20", "D1"}
    elif mat_model in ["gentthomas", "gt"]:
        true_params["C10"] = "1.0" if not (dev_p and len(dev_p) > 0) else f"{float(dev_p[0]):.4f}".rstrip('0').rstrip('.')
        # In Gent-Thomas, E is the 10th dev parameter (log term)
        e_val = dev_p[9] if (dev_p and len(dev_p) > 9) else 1.0
        true_params["E"] = f"{float(e_val):.4f}".rstrip('0').rstrip('.')
        true_params["D1"] = "1.5" if not (vol_p and len(vol_p) > 0) else f"{float(vol_p[0]):.4f}".rstrip('0').rstrip('.')
        true_active = {"C10", "E", "D1"}
    else:
        if dev_p and len(dev_p) > 0 and dev_p[0] != 0:
            true_params["C10"] = f"{float(dev_p[0]):.4f}".rstrip('0').rstrip('.')
            true_active.add("C10")
        if vol_p and len(vol_p) > 0 and vol_p[0] != 0:
            true_params["D1"] = f"{float(vol_p[0]):.4f}".rstrip('0').rstrip('.')
            true_active.add("D1")

    return true_params, true_active


def format_param_md(p_dict, true_val="-"):
    if not p_dict:
        val_str = "-"
    else:
        m = p_dict.get("mean")
        l = p_dict.get("95ci_lower")
        u = p_dict.get("95ci_upper")
        if m is None or np.isnan(m):
            val_str = "-"
        else:
            if 0 < abs(m) < 1e-3:
                val_str = f"{m:.2e} ({l:.2e}, {u:.2e})"
            else:
                val_str = f"{m:.4f} ({l:.4f}, {u:.4f})"
    if true_val != "-":
        return f"{val_str} [True: {true_val}]"
    return val_str


def format_param_tex(p_dict, true_val="-"):
    if not p_dict:
        val_str = "-"
    else:
        m = p_dict.get("mean")
        l = p_dict.get("95ci_lower")
        u = p_dict.get("95ci_upper")
        if m is None or np.isnan(m):
            val_str = "-"
        else:
            if 0 < abs(m) < 1e-3:
                val_str = f"{m:.2e} \\; ({l:.2e}, {u:.2e})"
            else:
                val_str = f"{m:.4f} \\; ({l:.4f}, {u:.4f})"
    if true_val != "-":
        return f"{val_str} \\; [\\text{{True: }} {true_val}]"
    return val_str


def main():
    args = parse_args()
    exp_dirs = [os.path.abspath(p) for p in args.exp_dirs]
    labels = args.labels or [os.path.basename(p).split("_")[1] if len(os.path.basename(p).split("_")) > 1 else os.path.basename(p) for p in exp_dirs]

    models_data = []
    for exp_dir, label in zip(exp_dirs, labels):
        config = load_experiment_config(exp_dir)
        mat_name = config.get("material_model_name", label)
        seed_dirs = find_seed_dirs(exp_dir)
        seed_list = []
        for s, p in seed_dirs:
            m, f = load_seed_metrics(p)
            if m is not None:
                seed_list.append({"seed": s, "path": p, "metrics": m})
        ranked = rank_seeds(seed_list)
        n_seeds = len(ranked)
        med_idx = n_seeds // 2
        med_seed = ranked[med_idx] if ranked else None
        best_seed = ranked[0] if ranked else None
        worst_seed = ranked[-1] if ranked else None
        true_params, true_active = get_true_model_info(config)

        models_data.append({
            "label": label,
            "mat_name": mat_name,
            "exp_dir": exp_dir,
            "config": config,
            "ranked_seeds": ranked,
            "n_seeds": n_seeds,
            "best_seed": best_seed,
            "med_seed": med_seed,
            "worst_seed": worst_seed,
            "true_params": true_params,
            "true_active": true_active,
        })

    # Metric Sections
    sections = [
        ("GP Posterior Extraction (SEF)", [
            ("GP $\\Psi$ RMSE", r"GP $\Psi$ RMSE", ("sef", "gp", "total", "rmse"), 4, False),
            ("GP $\\Psi$ EC (%)", r"GP $\Psi$ EC (\%)", ("sef", "gp", "total", "coverage"), 2, True),
            ("GP $\\Psi$ $R^2$", r"GP $\Psi$ $R^2$", ("sef", "gp", "total", "r2"), 4, False),
        ]),
        ("Validation Results (Distilled $\\Psi$)", [
            ("Distilled $\\Psi$ RMSE", r"Distilled $\Psi$ RMSE", ("sef", "dist", "total", "rmse"), 4, False),
            ("Distilled $\\Psi$ EC (%)", r"Distilled $\Psi$ EC (\%)", ("sef", "dist", "total", "coverage"), 2, True),
            ("Distilled $\\Psi$ $R^2$", r"Distilled $\Psi$ $R^2$", ("sef", "dist", "total", "r2"), 4, False),
        ]),
        ("Displacement Field Metrics (FEM)", [
            ("Disp RMSE (Block)", "Disp RMSE (Block)", ("disp", "block", "norm", "rmse"), 4, False),
            ("Disp EC (%) (Block)", r"Disp EC (\%) (Block)", ("disp", "block", "coverage_xy"), 2, True),
            ("Disp $R^2$ (Block)", r"Disp $R^2$ (Block)", ("disp", "block", "norm", "r2"), 4, False),
            ("Disp RMSE (Holes)", "Disp RMSE (Holes)", ("disp", "holes", "norm", "rmse"), 4, False),
            ("Disp EC (%) (Holes)", r"Disp EC (\%) (Holes)", ("disp", "holes", "coverage_xy"), 2, True),
            ("Disp $R^2$ (Holes)", r"Disp $R^2$ (Holes)", ("disp", "holes", "norm", "r2"), 4, False),
        ]),
        ("Conformal Calibration & Variance Budget", [
            ("Disp $Q_{0.95}$ (Calibrated on Block)", r"Disp $Q_{0.95}$ (Block)", ("conformal", "block", "q_disp"), 3, False),
            ("Calibrated Disp EC (%) (Block)", r"Calib Disp EC (\%) (Block)", ("conformal", "block", "calibrated_coverage_xy"), 2, True),
            ("Calibrated Disp EC (%) (Holes Transfer)", r"Calib Disp EC (\%) (Holes)", ("conformal", "holes", "calibrated_coverage_xy"), 2, True),
            ("Disp $\\sigma_{\\mathrm{param}}^2$", r"Disp $\sigma_{\mathrm{param}}^2$", ("conformal", "block", "variance_budget", "sigma2_param"), 4, False),
            ("Disp $\\sigma_{\\mathrm{DIC}}^2$", r"Disp $\sigma_{\mathrm{DIC}}^2$", ("conformal", "block", "variance_budget", "sigma2_noise"), 4, False),
            ("Disp $\\sigma_{\\mathrm{discrepancy}}^2$", r"Disp $\sigma_{\mathrm{discrepancy}}^2$", ("conformal", "block", "variance_budget", "sigma2_discrepancy"), 4, False),
            ("Disp Discrepancy Status", r"Disp Discrepancy Status", ("conformal", "block", "variance_budget", "status"), 0, False),
            ("Force $Q_{0.95}$ (Calibrated on Block)", r"Force $Q_{0.95}$ (Block)", ("force", "block", "conformal", "q_force"), 3, False),
            ("Calibrated Force EC (%) (Block)", r"Calib Force EC (\%) (Block)", ("force", "block", "conformal", "calibrated_total_ec"), 2, True),
            ("Calibrated Force EC (%) (Holes Transfer)", r"Calib Force EC (\%) (Holes)", ("force", "holes", "conformal", "calibrated_ec_y"), 2, True),
            ("Force $\\sigma_{\\mathrm{param}}^2$", r"Force $\sigma_{\mathrm{param}}^2$", ("force", "block", "conformal", "variance_budget", "sigma2_param"), 4, False),
            ("Force $\\sigma_{\\mathrm{loadcell}}^2$", r"Force $\sigma_{\mathrm{loadcell}}^2$", ("force", "block", "conformal", "variance_budget", "sigma2_noise"), 4, False),
            ("Force $\\sigma_{\\mathrm{discrepancy}}^2$", r"Force $\sigma_{\mathrm{discrepancy}}^2$", ("force", "block", "conformal", "variance_budget", "sigma2_discrepancy"), 4, False),
            ("Force Discrepancy Status", r"Force Discrepancy Status", ("force", "block", "conformal", "variance_budget", "status"), 0, False),
        ])
    ]

    # Material Parameter candidate rows
    param_rows = [
        ("C10", "$C_{10}$", r"$C_{10}$"),
        ("C01", "$C_{01}$", r"$C_{01}$"),
        ("C20", "$C_{20}$", r"$C_{20}$"),
        ("C11", "$C_{11}$", r"$C_{11}$"),
        ("C02", "$C_{02}$", r"$C_{02}$"),
        ("C30", "$C_{30}$", r"$C_{30}$"),
        ("C21", "$C_{21}$", r"$C_{21}$"),
        ("C12", "$C_{12}$", r"$C_{12}$"),
        ("C03", "$C_{03}$", r"$C_{03}$"),
        ("E", "$E$ (log term)", r"$E$ ($\ln(I_2/3)$)"),
        ("D1", "$D_1$", r"$D_1$"),
        ("D2", "$D_2$", r"$D_2$"),
        ("D3", "$D_3$", r"$D_3$"),
    ]

    # Format model column headers
    col_headers_md = ["Metric / Parameter"]
    col_headers_tex = [r"\textbf{Metric / Parameter}"]
    aligns_md = ["---"]

    display_names = {
        "nh2": "Neo-Hookean (`nh2`)",
        "neohookean": "Neo-Hookean (`nh2`)",
        "gentthomas": "Gent-Thomas (`gt`)",
        "gt": "Gent-Thomas (`gt`)",
        "isihara": "Isihara (`isihara`)"
    }
    display_names_tex = {
        "nh2": r"\textbf{Neo-Hookean (nh2)}",
        "neohookean": r"\textbf{Neo-Hookean (nh2)}",
        "gentthomas": r"\textbf{Gent-Thomas (gt)}",
        "gt": r"\textbf{Gent-Thomas (gt)}",
        "isihara": r"\textbf{Isihara (isihara)}"
    }

    for m in models_data:
        lbl = m["label"].lower()
        d_name = display_names.get(lbl, f"{m['label']} (`{m['mat_name']}`)")
        d_name_tex = display_names_tex.get(lbl, f"\\textbf{{{m['label']}}}")
        med_seed_idx = m["med_seed"]["seed"] if m["med_seed"] else "?"
        col_headers_md.append(f"{d_name}<br>($N={m['n_seeds']}$, Median Seed {med_seed_idx})")
        col_headers_tex.append(f"{d_name_tex} \\\\ ($N={m['n_seeds']}$, Med. S{med_seed_idx})")
        aligns_md.append(":---:")

    n_cols = len(col_headers_md)

    # ==============================================================================
    # 1. Build Markdown Table
    # ==============================================================================
    md_lines = []
    md_lines.append("# Cross-Model Discovery Performance & Identified Parameters\n")
    md_lines.append("## Overview Table\n")
    md_lines.append(f"Aggregate quantitative metrics across all seeds are reported as **Mean ± Std** ($N=20$). "
                    f"Candidate material parameters are reported for the **median ranked model** of each material class as **Mean (95% CI)** with the ground truth in brackets `[True: ...]`. Parameters pruned during discovery are denoted by `-`.\n")
    md_lines.append("| " + " | ".join(col_headers_md) + " |")
    md_lines.append("| " + " | ".join(aligns_md) + " |")

    # Add Metric Sections
    for sec_title, row_defs in sections:
        md_lines.append("| " + " | ".join([f"**{sec_title}**"] + [""] * (n_cols - 1)) + " |")
        for label, _, keys, decimals, is_pct in row_defs:
            row = [label]
            for m in models_data:
                agg = get_agg_stat(m["ranked_seeds"], *keys, decimals=decimals, is_pct=is_pct)
                row.append(agg)
            md_lines.append("| " + " | ".join(row) + " |")

    # Add Parameter Section
    md_lines.append("| " + " | ".join(["**Material Parameters (Median Model, 95% CI)**"] + [""] * (n_cols - 1)) + " |")
    for p_key, p_label_md, _ in param_rows:
        row = [p_label_md]
        for m in models_data:
            med_m = m["med_seed"]["metrics"].get("model_structure", {}) if m["med_seed"] else {}
            p_dict = med_m.get(p_key)
            t_val = m["true_params"].get(p_key, "-")
            row.append(format_param_md(p_dict, true_val=t_val))
        md_lines.append("| " + " | ".join(row) + " |")

    # ==============================================================================
    # 2. Build LaTeX Table
    # ==============================================================================
    tex_lines = []
    tex_lines.append("\n## LaTeX Table\n")
    tex_lines.append("```latex")
    tex_lines.append(r"\begin{table}[htbp]")
    tex_lines.append(r"\centering")
    tex_lines.append(r"\resizebox{\textwidth}{!}{")
    tex_lines.append(f"\\begin{{tabular}}{{l{'c' * (n_cols - 1)}}}")
    tex_lines.append(r"\toprule")
    tex_header = " & ".join([col_headers_tex[0]] + [f"\\makecell{{{h}}}" for h in col_headers_tex[1:]]) + r" \\"
    tex_lines.append(tex_header)
    tex_lines.append(r"\midrule")

    for sec_title, row_defs in sections:
        tex_lines.append(f"\\multicolumn{{{n_cols}}}{{l}}{{\\textbf{{{sec_title}}}}} \\\\")
        tex_lines.append(r"\midrule")
        for _, tex_label, keys, decimals, is_pct in row_defs:
            row = [tex_label]
            for m in models_data:
                agg = get_agg_stat_tex(m["ranked_seeds"], *keys, decimals=decimals, is_pct=is_pct)
                row.append(agg)
            tex_lines.append(" & ".join(row) + r" \\")
        tex_lines.append(r"\midrule")

    tex_lines.append(f"\\multicolumn{{{n_cols}}}{{l}}{{\\textbf{{Material Parameters (Median Model, 95\\% CI)}}}} \\\\")
    tex_lines.append(r"\midrule")
    for p_key, _, p_label_tex in param_rows:
        row = [p_label_tex]
        for m in models_data:
            med_m = m["med_seed"]["metrics"].get("model_structure", {}) if m["med_seed"] else {}
            p_dict = med_m.get(p_key)
            t_val = m["true_params"].get(p_key, "-")
            row.append(format_param_tex(p_dict, true_val=t_val))
        tex_lines.append(" & ".join(row) + r" \\")

    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabular}")
    tex_lines.append(r"}")
    tex_lines.append(r"\caption{Cross-Model Discovery Performance and Identified Parameters across All Seeds. "
                    r"Metrics report Mean $\pm$ Std ($N=20$). Material parameters report posterior Mean (95\% CI) "
                    r"for the median ranked seed model against ground-truth values.}")
    tex_lines.append(r"\label{tab:cross_model_summary}")
    tex_lines.append(r"\end{table}")
    tex_lines.append("```\n")

    # Model Visualizations Reference
    vis_lines = []
    vis_lines.append("## Plot Directory Organization\n")
    vis_lines.append("For both models, comprehensive visualizations have been structured into ranked subfolders:")
    for m in models_data:
        vis_lines.append(f"### {m['label']} (`{m['mat_name']}`)")
        vis_lines.append(f"- **Best Model (Rank 1, Seed {m['best_seed']['seed']})**: `{m['exp_dir']}/plots/best/`")
        vis_lines.append(f"- **Median Model (Rank {m['n_seeds']//2 + 1}, Seed {m['med_seed']['seed']})**: `{m['exp_dir']}/plots/median/`")
        vis_lines.append(f"- **Worst Model (Rank {m['n_seeds']}, Seed {m['worst_seed']['seed']})**: `{m['exp_dir']}/plots/worst/`\n")

    final_content = "\n".join(md_lines) + "\n" + "\n".join(tex_lines) + "\n" + "\n".join(vis_lines)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write(final_content)

    print(f"✅ Generated cross-model summary at: {args.out_md}")


if __name__ == "__main__":
    main()
