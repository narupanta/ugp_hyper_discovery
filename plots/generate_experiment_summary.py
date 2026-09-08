#!/usr/bin/env python3
import os
import sys
import glob
import json
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import yaml

from plots.theme import apply_style, save_figure, MODE_COLORS, COMPONENT_COLORS

# Default candidate parameters in GMR model discovery
ALL_CANDIDATE_PARAMS = ["C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03", "E", "D1", "D2", "D3"]
DEV_PARAMS_SET = {"C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03", "E"}
VOL_PARAMS_SET = {"D1", "D2", "D3"}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate cross-seed summary tables and plots for an experiment.")
    parser.add_argument("--experiment_dir", type=str, required=True, help="Path to the experiment folder under results/")
    parser.add_argument("--true_model", type=str, default=None, help="Ground truth material model name (e.g. nh2, isihara)")
    return parser.parse_args()


def load_experiment_config(exp_dir):
    cfg_path = os.path.join(exp_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(exp_dir, "recipe_config.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def find_seed_dirs(exp_dir):
    """Find all subdirectories whose names are integer seeds."""
    seed_dirs = []
    if not os.path.exists(exp_dir):
        return seed_dirs
    for entry in os.listdir(exp_dir):
        full_p = os.path.join(exp_dir, entry)
        if os.path.isdir(full_p) and entry.isdigit():
            seed_dirs.append((int(entry), full_p))
    seed_dirs.sort(key=lambda x: x[0])
    return seed_dirs


def load_seed_metrics(seed_path):
    """Locate and load validation_metrics.json for a given seed directory."""
    candidates = [
        os.path.join(seed_path, "fem_validation", "validation_metrics.json"),
        os.path.join(seed_path, "distilled", "validation_metrics.json"),
        os.path.join(seed_path, "validation_metrics.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                with open(c, "r") as f:
                    return json.load(f), c
            except Exception as e:
                print(f"Warning: Failed reading {c}: {e}")
    # Also check glob for validation_metrics_*.json
    for pattern in [
        os.path.join(seed_path, "fem_validation", "validation_metrics_*.json"),
        os.path.join(seed_path, "distilled", "validation_metrics_*.json"),
    ]:
        matches = glob.glob(pattern)
        if matches:
            try:
                with open(matches[0], "r") as f:
                    return json.load(f), matches[0]
            except Exception:
                pass
    return None, None


def rank_seeds(seed_data_list):
    """
    Rank seeds primarily by Block geometry FEM metrics:
    1. Higher Block R2 (norm)
    2. Lower Block RMSE (norm)
    3. Closeness of Coverage to 95%
    """
    def score_seed(item):
        m = item["metrics"]
        disp_b = m.get("disp", {}).get("block", {})
        norm_b = disp_b.get("norm", {})
        
        r2 = norm_b.get("r2", -999.0)
        rmse = norm_b.get("rmse", 999.0)
        cov = disp_b.get("coverage_xy", norm_b.get("coverage", 0.0))
        
        cov_dist = abs(cov - 95.0)
        return (r2, -rmse, -cov_dist)

    sorted_list = sorted(seed_data_list, key=score_seed, reverse=True)
    for rank, item in enumerate(sorted_list, start=1):
        item["rank"] = rank
    return sorted_list


def copy_best_seed_plots(best_seed_item, exp_dir, mat_model):
    """
    Copies key individual plots from the top-ranked seed into exp_dir/plots/:
    - split_energy_{mat_model}.pdf/.png
    - split_params_{mat_model}.pdf/.png
    - sobol_total_order_vs_invariants.pdf/.png
    - displacement_analysis.pdf/.png (for both block and holes)
    - disp_r2_coverage_xy__piola.pdf/.png (for both block and holes)
    """
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    seed_p = best_seed_item["path"]
    distilled_p = os.path.join(seed_p, "distilled")
    fem_p = os.path.join(seed_p, "fem_validation")

    # 1. Distillation plots
    for ext in ["pdf", "png"]:
        # split_energy
        for cand in [
            os.path.join(distilled_p, f"split_energy_{mat_model}.{ext}"),
            os.path.join(distilled_p, f"split_energy.{ext}"),
        ]:
            if os.path.exists(cand):
                shutil.copyfile(cand, os.path.join(plots_dir, f"split_energy_{mat_model}.{ext}"))
                break

        # split_params
        for cand in [
            os.path.join(distilled_p, f"split_params_{mat_model}.{ext}"),
            os.path.join(distilled_p, f"split_params.{ext}"),
        ]:
            if os.path.exists(cand):
                shutil.copyfile(cand, os.path.join(plots_dir, f"split_params_{mat_model}.{ext}"))
                break

        # sobol_total_order_vs_invariants
        for cand in [
            os.path.join(distilled_p, f"sobol_total_order_vs_invariants.{ext}"),
            os.path.join(distilled_p, "output", f"sobol_total_order_vs_invariants.{ext}"),
        ]:
            if os.path.exists(cand):
                shutil.copyfile(cand, os.path.join(plots_dir, f"sobol_total_order_vs_invariants.{ext}"))
                break

    # 2. FEM Validation plots for block and holes
    geom_sources = [
        ("block", [
            os.path.join(fem_p, "block"),
            os.path.join(fem_p),
            os.path.join(distilled_p, "fem_validation"),
        ]),
        ("holes", [
            os.path.join(fem_p, "holes"),
            os.path.join(fem_p, "fem_validation_holes"),
            os.path.join(distilled_p, "fem_validation_holes"),
        ]),
    ]

    for geom, paths in geom_sources:
        for ext in ["pdf", "png"]:
            # displacement_analysis
            for base in paths:
                cand = os.path.join(base, f"displacement_analysis.{ext}")
                if os.path.exists(cand):
                    shutil.copyfile(cand, os.path.join(plots_dir, f"displacement_analysis_{geom}.{ext}"))
                    break

            # disp_r2_coverage_xy__piola
            for base in paths:
                cand = os.path.join(base, f"disp_r2_coverage_xy__piola.{ext}")
                if os.path.exists(cand):
                    shutil.copyfile(cand, os.path.join(plots_dir, f"disp_r2_coverage_xy__piola_{geom}.{ext}"))
                    break


def generate_accuracy_comparison_plot(ranked_seeds, exp_dir):
    """Plots multi-seed accuracy bar chart (Block Disp R2, Holes Disp R2, SEF R2)."""
    apply_style()
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    seeds_num = [s["seed"] for s in ranked_seeds]
    ranks = [s["rank"] for s in ranked_seeds]

    disp_block_r2 = [
        s["metrics"].get("disp", {}).get("block", {}).get("norm", {}).get("r2", 0.0)
        for s in ranked_seeds
    ]
    disp_holes_r2 = [
        s["metrics"].get("disp", {}).get("holes", {}).get("norm", {}).get("r2", 0.0)
        for s in ranked_seeds
    ]
    sef_r2 = [
        s["metrics"].get("sef", {}).get("dist", {}).get("total", {}).get("r2", 0.0)
        for s in ranked_seeds
    ]

    x = np.arange(len(seeds_num))
    width = 0.26

    fig, ax = plt.subplots(figsize=(max(8, len(seeds_num) * 1.2), 4.8))

    ax.bar(x - width, disp_block_r2, width, label="Train Disp $R^2$ (Block)", color=MODE_COLORS.get("UT", "#1f77b4"), alpha=0.9)
    ax.bar(x, disp_holes_r2, width, label="Val Disp $R^2$ (Holes)", color=MODE_COLORS.get("PS", "#2ca02c"), alpha=0.9)
    ax.bar(x + width, sef_r2, width, label="SEF $R^2$ (Distilled)", color=COMPONENT_COLORS.get("dev", "#ff7f0e"), alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Rank {r}\n(Seed {s})" for r, s in zip(ranks, seeds_num)], fontsize=10)
    ax.set_ylabel(r"Coefficient of Determination ($R^2$)", fontsize=12)
    ax.set_title("Multi-Seed Model Accuracy (Ranked by Block FEM Performance)", fontsize=13, fontweight="bold")
    ax.legend(frameon=True, loc="lower left", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    min_val = min([v for v in disp_block_r2 + disp_holes_r2 if v > 0] or [0.9])
    ax.set_ylim([max(0.7, min_val - 0.05), 1.005])

    save_figure(fig, os.path.join(plots_dir, "seeds_accuracy_comparison.pdf"), make_png=True)
    plt.close(fig)





def clean_param_name(raw):
    return raw.replace('$', '').replace('{', '').replace('}', '').replace('_', '')


def generate_confusion_matrix_plot(ranked_seeds, exp_dir, config):
    """Plots active term selection and Sobol total-order sensitivity heatmap across seeds,
    matching the format of confusion_matrix_parameters.pdf."""
    import csv
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    true_params_info, true_active = get_true_model_info(config)

    # Sort seeds for x-axis: sort by seed number for clean index progression, or use ranked_seeds
    # generate_summary_tables_and_plots sorts seeds numerically by seed index:
    sorted_seeds = sorted(ranked_seeds, key=lambda x: x["seed"])

    # Load Sobol data for each seed
    sobol_data = []
    for s in sorted_seeds:
        seed_sobol = {}
        s_path = s["path"]
        # Look for dev sensitivities
        dev_candidates = [
            os.path.join(s_path, "distilled", "output", "dev_sensitivities", "total_sobol_indices_statistics_output_0.csv"),
            os.path.join(s_path, "distilled", "dev_sensitivities", "total_sobol_indices_statistics_output_0.csv"),
            os.path.join(s_path, "output", "dev_sensitivities", "total_sobol_indices_statistics_output_0.csv"),
        ]
        for dev_f in dev_candidates:
            if os.path.exists(dev_f):
                with open(dev_f, 'r') as f:
                    reader = list(csv.reader(f))
                    if len(reader) >= 2:
                        header = reader[0]
                        mean_row = reader[1]
                        for h, v in zip(header[1:], mean_row[1:]):
                            p_name = clean_param_name(h)
                            try:
                                seed_sobol[p_name] = float(v)
                            except ValueError:
                                pass
                break

        # Look for vol sensitivities
        vol_candidates = [
            os.path.join(s_path, "distilled", "output", "vol_sensitivities", "total_sobol_indices_statistics_output_0.csv"),
            os.path.join(s_path, "distilled", "vol_sensitivities", "total_sobol_indices_statistics_output_0.csv"),
            os.path.join(s_path, "output", "vol_sensitivities", "total_sobol_indices_statistics_output_0.csv"),
        ]
        for vol_f in vol_candidates:
            if os.path.exists(vol_f):
                with open(vol_f, 'r') as f:
                    reader = list(csv.reader(f))
                    if len(reader) >= 2:
                        header = reader[0]
                        mean_row = reader[1]
                        for h, v in zip(header[1:], mean_row[1:]):
                            p_name = clean_param_name(h)
                            try:
                                seed_sobol[p_name] = float(v)
                            except ValueError:
                                pass
                break

        sobol_data.append(seed_sobol)

    param_labels = [f"${p[0]}_{{{p[1:]}}}$" if len(p) > 1 else f"${p}$" for p in ALL_CANDIDATE_PARAMS]
    dev_params = set([p for p in ALL_CANDIDATE_PARAMS if p.startswith("C") or p == "E"])
    vol_params = set([p for p in ALL_CANDIDATE_PARAMS if p.startswith("D")])

    # Figure dimensions scaled nicely with seed count
    n_seeds = len(sorted_seeds)
    fig_w = max(7.0, min(14.0, 0.45 * n_seeds + 4.5))
    fig, ax = plt.subplots(figsize=(fig_w, 8.5), dpi=300)

    # RGBA Matrix
    rgba_matrix = np.ones((len(ALL_CANDIDATE_PARAMS), n_seeds, 4))
    rgba_matrix[:, :, :3] = 0.985  # light background for inactive cells

    cmap_dev = plt.cm.Blues
    cmap_vol = plt.cm.Oranges

    def get_color_intensity(val, is_dev=True):
        val_clamped = max(1e-5, min(1.0, val))
        intensity = 0.12 + 0.46 * (val_clamped ** 0.35)
        cmap = cmap_dev if is_dev else cmap_vol
        return cmap(intensity)

    for j, s in enumerate(sorted_seeds):
        terms = set(s["metrics"].get("model_structure", {}).keys())
        seed_s = sobol_data[j]

        for i, p in enumerate(ALL_CANDIDATE_PARAMS):
            if p in terms:
                sens = seed_s.get(p, 0.0)
                is_dev = p in dev_params
                color = get_color_intensity(sens, is_dev=is_dev)
                rgba_matrix[i, j] = color

                if sens >= 0.1:
                    txt = f"{sens:.2f}"
                elif sens >= 0.001:
                    txt = f"{sens:.3f}"
                else:
                    txt = "<0.001"

                text_color = "#111111"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                        fontweight="bold" if p in true_active else "normal",
                        color=text_color)
            else:
                ax.text(j, i, "·", ha="center", va="center", color="#cccccc", fontsize=9)

    ax.imshow(rgba_matrix, aspect='auto')

    ax.set_yticks(np.arange(len(ALL_CANDIDATE_PARAMS)))
    ax.set_yticklabels(param_labels, fontsize=12)
    ax.set_xticks(np.arange(n_seeds))
    ax.set_xticklabels([f"{s['seed']}" for s in sorted_seeds], fontsize=10.5)

    ax.set_xlabel("Random Seed Index", fontsize=12.5, labelpad=8, fontweight='bold')
    ax.set_ylabel("Candidate Constitutive Parameters", fontsize=12.5, labelpad=8, fontweight='bold')

    # Highlight ground truth active parameters with circle
    for tick_idx, p in enumerate(ALL_CANDIDATE_PARAMS):
        if p in true_active:
            ax.get_yticklabels()[tick_idx].set_color('#111111')
            ax.get_yticklabels()[tick_idx].set_fontweight('bold')
            ax.get_yticklabels()[tick_idx].set_bbox(dict(boxstyle="circle,pad=0.22", edgecolor="#111111", facecolor="none", linewidth=1.3))
        else:
            ax.get_yticklabels()[tick_idx].set_color('#333333')

    # Minor grid lines to separate cells
    ax.set_xticks(np.arange(-0.5, n_seeds, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ALL_CANDIDATE_PARAMS), 1), minor=True)
    ax.grid(which='minor', color='#d5d5d5', linestyle='-', linewidth=0.8)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.tick_params(which='major', length=0)

    # Separator line between deviatoric and volumetric candidate terms
    sep_idx = len(dev_params) - 0.5
    ax.axhline(sep_idx, color='#555555', linestyle='--', linewidth=1.5, alpha=0.7)

    # Custom Dual Colorbar / Legend
    cax_dev = fig.add_axes([0.16, 0.045, 0.30, 0.028])
    cax_vol = fig.add_axes([0.56, 0.045, 0.30, 0.028])

    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap_dev_cb = mcolors.LinearSegmentedColormap.from_list('Blues_cb', [plt.cm.Blues(x) for x in np.linspace(0.0, 0.58, 256)])
    sm_dev = plt.cm.ScalarMappable(cmap=cmap_dev_cb, norm=norm)
    sm_dev.set_array([])
    cb_dev = fig.colorbar(sm_dev, cax=cax_dev, orientation='horizontal')
    cb_dev.set_label(r'Deviatoric Sensitivity $\bar{S}_\mathrm{T}$ ($C_{ij}$, $E$)', fontsize=10, fontweight='bold', color='#08519c', labelpad=5)
    cb_dev.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cb_dev.ax.tick_params(labelsize=8.5)

    cmap_vol_cb = mcolors.LinearSegmentedColormap.from_list('Oranges_cb', [plt.cm.Oranges(x) for x in np.linspace(0.0, 0.58, 256)])
    sm_vol = plt.cm.ScalarMappable(cmap=cmap_vol_cb, norm=norm)
    sm_vol.set_array([])
    cb_vol = fig.colorbar(sm_vol, cax=cax_vol, orientation='horizontal')
    cb_vol.set_label(r'Volumetric Sensitivity $\bar{S}_\mathrm{T}$ ($D_i$ terms)', fontsize=10, fontweight='bold', color='#a63603', labelpad=5)
    cb_vol.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cb_vol.ax.tick_params(labelsize=8.5)

    plt.subplots_adjust(bottom=0.18, top=0.97, left=0.11, right=0.97)

    cm_pdf = os.path.join(plots_dir, "confusion_matrix_parameters.pdf")
    cm_png = os.path.join(plots_dir, "confusion_matrix_parameters.png")
    plt.savefig(cm_png, dpi=300, bbox_inches='tight')
    plt.savefig(cm_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Generated sensitivity-aware confusion matrix at {cm_pdf}")


def get_true_model_info(config):
    mat_model = config.get("material_model_name", "nh2").lower()
    true_params = {p: "-" for p in ALL_CANDIDATE_PARAMS}
    true_active = set()

    mat_p = config.get("material_params", {})
    dev_p = mat_p.get("dev_params", config.get("dev_params", None))
    vol_p = mat_p.get("vol_params", config.get("vol_params", None))

    if mat_model == "nh2":
        true_params["C10"] = "0.5" if not (dev_p and len(dev_p) > 0) else f"{float(dev_p[0]):.4f}".rstrip('0').rstrip('.')
        true_params["D1"] = "1.5" if not (vol_p and len(vol_p) > 0) else f"{float(vol_p[0]):.4f}".rstrip('0').rstrip('.')
        true_active = {"C10", "D1"}
    elif mat_model == "isihara":
        true_params["C10"] = "0.5" if not (dev_p and len(dev_p) > 0) else f"{float(dev_p[0]):.4f}".rstrip('0').rstrip('.')
        true_params["C01"] = "0.1" if not (dev_p and len(dev_p) > 1) else f"{float(dev_p[1]):.4f}".rstrip('0').rstrip('.')
        true_params["C20"] = "0.05" if not (dev_p and len(dev_p) > 2) else f"{float(dev_p[2]):.4f}".rstrip('0').rstrip('.')
        true_params["D1"] = "1.5" if not (vol_p and len(vol_p) > 0) else f"{float(vol_p[0]):.4f}".rstrip('0').rstrip('.')
        true_active = {"C10", "C01", "C20", "D1"}
    elif mat_model == "gentthomas":
        true_params["C10"] = "0.5" if not (dev_p and len(dev_p) > 0) else f"{float(dev_p[0]):.4f}".rstrip('0').rstrip('.')
        true_params["C01"] = "0.1" if not (dev_p and len(dev_p) > 1) else f"{float(dev_p[1]):.4f}".rstrip('0').rstrip('.')
        true_params["D1"] = "1.5" if not (vol_p and len(vol_p) > 0) else f"{float(vol_p[0]):.4f}".rstrip('0').rstrip('.')
        true_active = {"C10", "C01", "D1"}
    else:
        if dev_p and len(dev_p) > 0 and dev_p[0] != 0:
            true_params["C10"] = f"{float(dev_p[0]):.4f}".rstrip('0').rstrip('.')
            true_active.add("C10")
        if vol_p and len(vol_p) > 0 and vol_p[0] != 0:
            true_params["D1"] = f"{float(vol_p[0]):.4f}".rstrip('0').rstrip('.')
            true_active.add("D1")

    return true_params, true_active


def fmt_val(v, decimals=4):
    if v is None or v == "N/A":
        return "-"
    try:
        val = float(v)
        if np.isnan(val):
            return "-"
        return f"{val:.{decimals}f}"
    except (ValueError, TypeError):
        return str(v)


def format_param_md(p_dict):
    if not p_dict:
        return "-"
    m = p_dict.get("mean")
    l = p_dict.get("95ci_lower")
    u = p_dict.get("95ci_upper")
    if m is None or np.isnan(m):
        return "-"
    return f"{m:.4f}<br>({l:.4f}, {u:.4f})"


def format_param_latex(p_dict):
    if not p_dict:
        return "-"
    m = p_dict.get("mean")
    l = p_dict.get("95ci_lower")
    u = p_dict.get("95ci_upper")
    if m is None or np.isnan(m):
        return "-"
    return f"\\begin{{tabular}}{{@{{}}c@{{}}}}{m:.4f}\\\\({l:.4f}, {u:.4f})\\end{{tabular}}"


def format_summary_markdown(ranked_seeds, exp_dir, config):
    """Generates summary_across_seeds.md matching the layout of results_table_top10.md."""
    out_md = os.path.join(exp_dir, "summary_across_seeds.md")

    mat_model = config.get("material_model_name", "nh2")
    top_n = min(10, len(ranked_seeds))
    top_seeds = ranked_seeds[:top_n]
    true_params, true_active = get_true_model_info(config)

    # LaTeX parameter display map
    param_math = {
        "C10": "$C_{10}$", "C01": "$C_{01}$", "C20": "$C_{20}$", "C11": "$C_{11}$",
        "C02": "$C_{02}$", "C30": "$C_{30}$", "C21": "$C_{21}$", "C12": "$C_{12}$",
        "C03": "$C_{03}$", "E": "$E$", "D1": "$D_{1}$", "D2": "$D_{2}$", "D3": "$D_{3}$"
    }

    # Precompute structure discovery metrics for each seed
    structure_metrics = []
    for s in top_seeds:
        struct = s["metrics"].get("model_structure", {})
        pred_active = set(struct.keys())
        tp = len(pred_active & true_active)
        fp = len(pred_active - true_active)
        fn = len(true_active - pred_active)
        prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if len(true_active) == 0 else 0.0)
        rec = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if len(true_active) == 0 else 0.0)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        structure_metrics.append({
            "prec": prec,
            "rec": rec,
            "f1": f1
        })

    # ==============================================================================
    # 1. Markdown Table Generation
    # ==============================================================================
    col_headers = ["Metric / Parameter"] + [f"Seed {s['seed']}" for s in top_seeds] + ["True Value"]
    aligns = ["---"] + [":---:"] * (len(top_seeds) + 1)
    n_cols = len(col_headers)

    md_lines = []
    md_lines.append(f"# Results (Top {top_n} Seeds Ranked by Validation Disp R²)\n")
    md_lines.append("## Markdown Table\n")
    md_lines.append("| " + " | ".join(col_headers) + " |")
    md_lines.append("| " + " | ".join(aligns) + " |")

    def add_sec_hdr_md(title):
        row = [f"**{title}**"] + [""] * (n_cols - 1)
        md_lines.append("| " + " | ".join(row) + " |")

    def add_row_md(name, values, true_v="-"):
        row = [name] + [str(v) for v in values] + [str(true_v)]
        md_lines.append("| " + " | ".join(row) + " |")

    # Section 1: GP Posterior Extraction
    add_sec_hdr_md("GP Posterior Extraction")
    add_row_md("GP $\\Psi$ RMSE", [fmt_val(s["metrics"].get("sef", {}).get("gp", {}).get("total", {}).get("rmse")) for s in top_seeds])
    add_row_md("GP $\\Psi$ EC (%)", [fmt_val(s["metrics"].get("sef", {}).get("gp", {}).get("total", {}).get("coverage"), 2) for s in top_seeds])
    add_row_md("GP $\\Psi$ $R^2$", [fmt_val(s["metrics"].get("sef", {}).get("gp", {}).get("total", {}).get("r2")) for s in top_seeds])

    # Section 2: Validation Results (Distilled Psi)
    add_sec_hdr_md("Validation Results (Distilled $\\Psi$)")
    add_row_md("Distilled $\\Psi$ RMSE", [fmt_val(s["metrics"].get("sef", {}).get("dist", {}).get("total", {}).get("rmse")) for s in top_seeds])
    add_row_md("Distilled $\\Psi$ EC (%)", [fmt_val(s["metrics"].get("sef", {}).get("dist", {}).get("total", {}).get("coverage"), 2) for s in top_seeds])
    add_row_md("Distilled $\\Psi$ $R^2$", [fmt_val(s["metrics"].get("sef", {}).get("dist", {}).get("total", {}).get("r2")) for s in top_seeds])

    # Section 3: Displacement Field Metrics (FEM)
    add_sec_hdr_md("Displacement Field Metrics (FEM)")
    add_row_md("Disp RMSE (Holes)", [fmt_val(s["metrics"].get("disp", {}).get("holes", {}).get("norm", {}).get("rmse")) for s in top_seeds])
    add_row_md("Disp EC (%) (Holes)", [fmt_val(s["metrics"].get("disp", {}).get("holes", {}).get("coverage_xy", s["metrics"].get("disp", {}).get("holes", {}).get("norm", {}).get("coverage")), 2) for s in top_seeds])
    add_row_md("Disp $R^2$ (Holes)", [fmt_val(s["metrics"].get("disp", {}).get("holes", {}).get("norm", {}).get("r2")) for s in top_seeds])
    add_row_md("Disp RMSE (Block)", [fmt_val(s["metrics"].get("disp", {}).get("block", {}).get("norm", {}).get("rmse")) for s in top_seeds])
    add_row_md("Disp EC (%) (Block)", [fmt_val(s["metrics"].get("disp", {}).get("block", {}).get("coverage_xy", s["metrics"].get("disp", {}).get("block", {}).get("norm", {}).get("coverage")), 2) for s in top_seeds])
    add_row_md("Disp $R^2$ (Block)", [fmt_val(s["metrics"].get("disp", {}).get("block", {}).get("norm", {}).get("r2")) for s in top_seeds])

    # Section 4: Model Structure Discovery
    add_sec_hdr_md("Model Structure Discovery (UQ / Pruning)")
    add_row_md("Term Precision", [fmt_val(sm["prec"], 3) for sm in structure_metrics], "1.000")
    add_row_md("Term Recall", [fmt_val(sm["rec"], 3) for sm in structure_metrics], "1.000")
    add_row_md("Term F1-Score", [fmt_val(sm["f1"], 3) for sm in structure_metrics], "1.000")

    # Section 5: Discovered Model Parameters
    add_sec_hdr_md("Discovered Model Parameters")
    for p in ALL_CANDIDATE_PARAMS:
        label = param_math[p]
        p_cells = [format_param_md(s["metrics"].get("model_structure", {}).get(p)) for s in top_seeds]
        true_v = true_params.get(p, "-")
        add_row_md(label, p_cells, true_v)

    # ==============================================================================
    # 2. LaTeX Table Generation
    # ==============================================================================
    tex_lines = []
    tex_col_spec = "l" + "c" * len(top_seeds) + "c"
    tex_header = " & ".join([r"\textbf{Metric / Parameter}"] + [f"\\textbf{{Seed {s['seed']}}}" for s in top_seeds] + [r"\textbf{True Value}"]) + r" \\"

    tex_lines.append("\n## LaTeX Table\n")
    tex_lines.append("```latex")
    tex_lines.append(r"\begin{table}[htbp]")
    tex_lines.append(r"\centering")
    tex_lines.append(r"\resizebox{\textwidth}{!}{")
    tex_lines.append(f"\\begin{{tabular}}{{{tex_col_spec}}}")
    tex_lines.append(r"\toprule")
    tex_lines.append(tex_header)
    tex_lines.append(r"\midrule")

    def add_sec_hdr_tex(title):
        tex_lines.append(f"\\multicolumn{{{n_cols}}}{{l}}{{\\textbf{{{title}}}}} \\\\")
        tex_lines.append(r"\midrule")

    def add_row_tex(name, values, true_v="-"):
        tex_row = " & ".join([name] + [str(v) for v in values] + [str(true_v)]) + r" \\"
        tex_lines.append(tex_row)

    add_sec_hdr_tex("GP Posterior Extraction")
    add_row_tex(r"GP $\Psi$ RMSE", [fmt_val(s["metrics"].get("sef", {}).get("gp", {}).get("total", {}).get("rmse")) for s in top_seeds])
    add_row_tex(r"GP $\Psi$ EC (\%)", [fmt_val(s["metrics"].get("sef", {}).get("gp", {}).get("total", {}).get("coverage"), 2) for s in top_seeds])
    add_row_tex(r"GP $\Psi$ $R^2$", [fmt_val(s["metrics"].get("sef", {}).get("gp", {}).get("total", {}).get("r2")) for s in top_seeds])
    tex_lines.append(r"\midrule")

    add_sec_hdr_tex(r"Validation Results (Distilled $\Psi$)")
    add_row_tex(r"Distilled $\Psi$ RMSE", [fmt_val(s["metrics"].get("sef", {}).get("dist", {}).get("total", {}).get("rmse")) for s in top_seeds])
    add_row_tex(r"Distilled $\Psi$ EC (\%)", [fmt_val(s["metrics"].get("sef", {}).get("dist", {}).get("total", {}).get("coverage"), 2) for s in top_seeds])
    add_row_tex(r"Distilled $\Psi$ $R^2$", [fmt_val(s["metrics"].get("sef", {}).get("dist", {}).get("total", {}).get("r2")) for s in top_seeds])
    tex_lines.append(r"\midrule")

    add_sec_hdr_tex("Displacement Field Metrics (FEM)")
    add_row_tex("Disp RMSE (Holes)", [fmt_val(s["metrics"].get("disp", {}).get("holes", {}).get("norm", {}).get("rmse")) for s in top_seeds])
    add_row_tex(r"Disp EC (\%) (Holes)", [fmt_val(s["metrics"].get("disp", {}).get("holes", {}).get("coverage_xy", s["metrics"].get("disp", {}).get("holes", {}).get("norm", {}).get("coverage")), 2) for s in top_seeds])
    add_row_tex(r"Disp $R^2$ (Holes)", [fmt_val(s["metrics"].get("disp", {}).get("holes", {}).get("norm", {}).get("r2")) for s in top_seeds])
    add_row_tex("Disp RMSE (Block)", [fmt_val(s["metrics"].get("disp", {}).get("block", {}).get("norm", {}).get("rmse")) for s in top_seeds])
    add_row_tex(r"Disp EC (\%) (Block)", [fmt_val(s["metrics"].get("disp", {}).get("block", {}).get("coverage_xy", s["metrics"].get("disp", {}).get("block", {}).get("norm", {}).get("coverage")), 2) for s in top_seeds])
    add_row_tex(r"Disp $R^2$ (Block)", [fmt_val(s["metrics"].get("disp", {}).get("block", {}).get("norm", {}).get("r2")) for s in top_seeds])
    tex_lines.append(r"\midrule")

    add_sec_hdr_tex("Model Structure Discovery Metrics")
    add_row_tex("Term Precision", [fmt_val(sm["prec"], 3) for sm in structure_metrics], "1.000")
    add_row_tex("Term Recall", [fmt_val(sm["rec"], 3) for sm in structure_metrics], "1.000")
    add_row_tex("Term F1-Score", [fmt_val(sm["f1"], 3) for sm in structure_metrics], "1.000")
    tex_lines.append(r"\midrule")

    add_sec_hdr_tex("Discovered Model Parameters")
    for p in ALL_CANDIDATE_PARAMS:
        label = param_math[p]
        p_cells = [format_param_latex(s["metrics"].get("model_structure", {}).get(p)) for s in top_seeds]
        true_v = true_params.get(p, "-")
        add_row_tex(label, p_cells, true_v)

    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabular}")
    tex_lines.append(r"}")
    tex_lines.append(f"\\caption{{Distillation Results and Discovered Material Parameters (Top {top_n} Seeds Ranked by Validation Disp R²)}}")
    tex_lines.append(f"\\label{{tab:distillation_results_{top_n}}}")
    tex_lines.append(r"\end{table}")
    tex_lines.append("```\n")

    # ==============================================================================
    # 3. Best-Seed Visualizations Links
    # ==============================================================================
    vis_lines = []
    vis_lines.append("## Key Visualizations (Top Ranked Seed)\n")
    vis_lines.append(f"- **Strain Energy Splitting**: `plots/split_energy_{mat_model}.pdf`")
    vis_lines.append(f"- **Candidate Parameters Posterior**: `plots/split_params_{mat_model}.pdf`")
    vis_lines.append(f"- **Sobol Sensitivity Analysis**: `plots/sobol_total_order_vs_invariants.pdf`")
    vis_lines.append(f"- **Block Displacement UQ**: `plots/displacement_analysis_block.pdf` & `plots/disp_r2_coverage_xy__piola_block.pdf`")
    vis_lines.append(f"- **Holes Displacement UQ**: `plots/displacement_analysis_holes.pdf` & `plots/disp_r2_coverage_xy__piola_holes.pdf`\n")

    final_content = "\n".join(md_lines) + "\n" + "\n".join(tex_lines) + "\n" + "\n".join(vis_lines)

    with open(out_md, "w") as f:
        f.write(final_content)

    print(f"✅ Generated summary markdown matching results_table_top10.md at: {out_md}")


def main():
    args = parse_args()
    exp_dir = os.path.abspath(args.experiment_dir)
    if not os.path.isdir(exp_dir):
        print(f"Error: Experiment directory '{exp_dir}' not found.")
        sys.exit(1)

    config = load_experiment_config(exp_dir)
    mat_model = args.true_model or config.get("material_model_name", "nh2")

    seed_dirs = find_seed_dirs(exp_dir)
    if not seed_dirs:
        print(f"No numeric seed directories found in {exp_dir}.")
        sys.exit(0)

    print(f"Found {len(seed_dirs)} seed directories in {exp_dir}...")

    seed_data_list = []
    for seed, s_path in seed_dirs:
        metrics, m_file = load_seed_metrics(s_path)
        if metrics is not None:
            seed_data_list.append({
                "seed": seed,
                "path": s_path,
                "metrics": metrics,
                "metrics_file": m_file,
            })
        else:
            print(f"Warning: No valid validation_metrics.json found for seed {seed} in {s_path}")

    if not seed_data_list:
        print("No completed seed metrics available to summarize.")
        sys.exit(0)

    # Rank seeds
    ranked_seeds = rank_seeds(seed_data_list)
    best_seed = ranked_seeds[0]
    print(f"🏆 Top Ranked Seed: Seed {best_seed['seed']} (Rank 1)")

    # Copy top seed plots into exp_dir/plots/
    copy_best_seed_plots(best_seed, exp_dir, mat_model)

    # Multi-seed plots
    generate_accuracy_comparison_plot(ranked_seeds, exp_dir)
    generate_confusion_matrix_plot(ranked_seeds, exp_dir, config)

    # Generate Markdown summary table
    format_summary_markdown(ranked_seeds, exp_dir, config)


if __name__ == "__main__":
    main()
