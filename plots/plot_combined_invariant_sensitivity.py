"""
plots/plot_combined_invariant_sensitivity.py: Plots combined invariant-dependent Sobol indices:
3 rows x 1 col layout:
1. Total Sobol order vs I1_bar (dev parameters only)
2. Total Sobol order vs I2_bar (dev parameters only)
3. Total Sobol order vs J (vol parameters only)
Merged unified legend at bottom outside the plots.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plots.theme import apply_style, save_figure
from plots.sensitivity import load_sobol_csv
from core.utils import compute_invariants_np, load_f3x3_from_distilled


def to_latex(name):
    clean = name.replace("$", "").replace("{", "").replace("}", "").replace("_", "")
    if clean.startswith("C") and len(clean) == 3 and clean[1:].isdigit():
        return rf"$C_{{{clean[1:]}}}$"
    if clean.startswith("D") and len(clean) == 2 and clean[1:].isdigit():
        return rf"$D_{{{clean[1:]}}}$"
    if clean == "E":
        return r"$E$"
    return rf"${clean}$"


def get_param_color(name):
    clean = name.replace("$", "").replace("{", "").replace("}", "").replace("_", "")
    if clean == "C10":
        return "#0072B2"
    elif clean == "C01":
        return "#56B4E9"
    elif clean == "D1":
        return "#D55E00"
    elif clean == "D2":
        return "#E69F00"
    elif clean.startswith("C4") or clean.startswith("C6"):
        return "#CC79A7"
    return "#009E73"


def main():
    apply_style()
    parser = argparse.ArgumentParser(description="Plot combined invariant-dependent Sobol indices (dev vs I1, I2; vol vs J)")
    parser.add_argument("--distilled_dir", type=str, required=True, help="Path to distilled model directory")
    parser.add_argument("--threshold", type=float, default=1e-4, help="Threshold to filter active parameters")
    parser.add_argument("--output_name", type=str, default="sobol_total_order_vs_invariants.pdf")
    args = parser.parse_args()

    distilled_dir = os.path.abspath(args.distilled_dir)
    if not os.path.exists(os.path.join(distilled_dir, "f3x3.npy")) and os.path.isdir(os.path.join(distilled_dir, "distilled")):
        distilled_dir = os.path.join(distilled_dir, "distilled")

    # 1. Load f3x3 deformation gradients and compute invariants
    f3x3 = load_f3x3_from_distilled(distilled_dir)
    if f3x3 is None:
        raise FileNotFoundError(f"Could not locate f3x3.npy for {distilled_dir}")

    I1_bar, I2_bar, J = compute_invariants_np(f3x3)
    n_points = len(f3x3)

    # 2. Locate Sobol CSV files
    df_dev, _ = load_sobol_csv(distilled_dir, "dev_sensitivities")
    if df_dev is None:
        df_dev, _ = load_sobol_csv(distilled_dir, "sensitivities")

    df_vol, _ = load_sobol_csv(distilled_dir, "vol_sensitivities")

    if df_dev is None or df_vol is None:
        missing = []
        if df_dev is None:
            missing.append("dev")
        if df_vol is None:
            missing.append("vol")
        print(f"Skipping combined invariant sensitivity plot: missing {', '.join(missing)} sensitivity results in {distilled_dir}")
        return

    df_dev = df_dev.iloc[:n_points]
    df_vol = df_vol.iloc[:n_points]

    exclude_cols = ['test cases', 'Unnamed: 0']
    dev_params = [c for c in df_dev.columns if c not in exclude_cols and not c.startswith('Unnamed')]
    vol_params = [c for c in df_vol.columns if c not in exclude_cols and not c.startswith('Unnamed')]

    active_dev = [p for p in dev_params if df_dev[p].max() > args.threshold]
    active_vol = [p for p in vol_params if df_vol[p].max() > args.threshold]

    if not active_dev:
        active_dev = [dev_params[0]]
    if not active_vol:
        active_vol = [vol_params[0]]

    # 3. Create 3 rows x 1 col plot for narrow column layout (ratio ~ 0.20)
    fig_width = 1.65
    fig_height = 4.30
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(3, 1, hspace=0.36, top=0.98, bottom=0.12, left=0.28, right=0.94)

    # Panel 1: Total Sobol Order vs I1_bar (Deviatoric parameters only)
    ax1 = fig.add_subplot(gs[0, 0])
    for j, param in enumerate(active_dev):
        color = get_param_color(param)
        ax1.scatter(I1_bar, df_dev[param].values, color=color, alpha=0.65, s=8, edgecolors='none')
    ax1.set_xlabel(r"$\bar{I}_1$", fontsize=6.8, labelpad=1)
    ax1.set_ylabel(r"$\bar{S}_{\mathrm{T,d}}$", fontsize=7.0, labelpad=1)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_yticks([0.0, 0.5, 1.0])
    ax1.tick_params(axis='both', which='major', labelsize=5.5, pad=1)
    ax1.grid(False)

    # Panel 2: Total Sobol Order vs I2_bar (Deviatoric parameters only)
    ax2 = fig.add_subplot(gs[1, 0])
    for j, param in enumerate(active_dev):
        color = get_param_color(param)
        ax2.scatter(I2_bar, df_dev[param].values, color=color, alpha=0.65, s=8, edgecolors='none')
    ax2.set_xlabel(r"$\bar{I}_2$", fontsize=6.8, labelpad=1)
    ax2.set_ylabel(r"$\bar{S}_{\mathrm{T,d}}$", fontsize=7.0, labelpad=1)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_yticks([0.0, 0.5, 1.0])
    ax2.tick_params(axis='both', which='major', labelsize=5.5, pad=1)
    ax2.grid(False)

    # Panel 3: Total Sobol Order vs J (Volumetric parameters only)
    ax3 = fig.add_subplot(gs[2, 0])
    for j, param in enumerate(active_vol):
        color = get_param_color(param)
        ax3.scatter(J, df_vol[param].values, color=color, alpha=0.65, s=8, edgecolors='none')
    ax3.set_xlabel(r"$J$", fontsize=6.8, labelpad=1)
    ax3.set_ylabel(r"$\bar{S}_{\mathrm{T,v}}$", fontsize=7.0, labelpad=1)
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_yticks([0.0, 0.5, 1.0])
    ax3.tick_params(axis='both', which='major', labelsize=5.5, pad=1)
    ax3.grid(False)

    # Unified bottom legend outside the plots (e.g. C10, C01, D1)
    all_active = list(dict.fromkeys(active_dev + active_vol))
    legend_handles = []
    legend_labels = []
    for p in all_active:
        color = get_param_color(p)
        legend_handles.append(mlines.Line2D([], [], color=color, marker='o', linestyle='none', markersize=4))
        legend_labels.append(to_latex(p))

    fig.legend(legend_handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, 0.01),
               ncol=len(legend_handles), fontsize=6.0, frameon=False, handletextpad=0.2, columnspacing=0.5)

    out_pdf = os.path.join(distilled_dir, args.output_name)
    base_name = os.path.splitext(args.output_name)[0]
    out_png = os.path.join(distilled_dir, f"{base_name}.png")

    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved combined invariant sensitivity plot to:\n  - {out_pdf}\n  - {out_png}")


if __name__ == "__main__":
    main()
