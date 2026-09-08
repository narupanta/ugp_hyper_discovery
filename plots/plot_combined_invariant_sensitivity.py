"""
plots/plot_combined_invariant_sensitivity.py: Plots combined invariant-dependent Sobol indices:
1. Total Sobol order vs I1_bar (dev parameters only)
2. Total Sobol order vs I2_bar (dev parameters only)
3. Total Sobol order vs J (vol parameters only)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plots.theme import apply_style, save_figure
from plots.sensitivity import load_sobol_csv
from core.utils import compute_invariants_np, load_f3x3_from_distilled


def main():
    apply_style()
    parser = argparse.ArgumentParser(description="Plot combined invariant-dependent Sobol indices (dev vs I1, I2; vol vs J)")
    parser.add_argument("--distilled_dir", type=str, required=True, help="Path to distilled model directory")
    parser.add_argument("--threshold", type=float, default=1e-4, help="Threshold to filter active parameters")
    parser.add_argument("--output_name", type=str, default="sobol_total_order_vs_invariants.pdf")
    args = parser.parse_args()

    distilled_dir = os.path.abspath(args.distilled_dir)

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

    # 3. Create 3-panel plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    tab10_colors = plt.cm.tab10.colors
    dev_colors = [tab10_colors[i % len(tab10_colors)] for i in range(len(active_dev))]
    vol_colors = [tab10_colors[(len(active_dev) + i) % len(tab10_colors)] for i in range(len(active_vol))]

    # Panel 1: Total Sobol Order vs I1_bar (Deviatoric parameters only)
    ax1 = axes[0]
    for j, param in enumerate(active_dev):
        label_clean = param.replace("$", "")
        ax1.scatter(I1_bar, df_dev[param].values, label=f"${label_clean}$",
                    color=dev_colors[j], alpha=0.75, s=26, edgecolors='none')
    ax1.set_title(r"Total Sobol Order vs $\bar{I}_1$", fontsize=13)
    ax1.set_xlabel(r"$\bar{I}_1$", fontsize=12)
    ax1.set_ylabel(r"Total-Order Sobol Index ($S_T$)", fontsize=12)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best", fontsize=11, framealpha=0.92)

    # Panel 2: Total Sobol Order vs I2_bar (Deviatoric parameters only)
    ax2 = axes[1]
    for j, param in enumerate(active_dev):
        label_clean = param.replace("$", "")
        ax2.scatter(I2_bar, df_dev[param].values, label=f"${label_clean}$",
                    color=dev_colors[j], alpha=0.75, s=26, edgecolors='none')
    ax2.set_title(r"Total Sobol Order vs $\bar{I}_2$", fontsize=13)
    ax2.set_xlabel(r"$\bar{I}_2$", fontsize=12)
    ax2.set_ylabel(r"Total-Order Sobol Index ($S_T$)", fontsize=12)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best", fontsize=11, framealpha=0.92)

    # Panel 3: Total Sobol Order vs J (Volumetric parameters only)
    ax3 = axes[2]
    for j, param in enumerate(active_vol):
        label_clean = param.replace("$", "")
        ax3.scatter(J, df_vol[param].values, label=f"${label_clean}$",
                    color=vol_colors[j], alpha=0.75, s=26, edgecolors='none')
    ax3.set_title(r"Total Sobol Order vs $J$", fontsize=13)
    ax3.set_xlabel(r"$J$", fontsize=12)
    ax3.set_ylabel(r"Total-Order Sobol Index ($S_T$)", fontsize=12)
    ax3.set_ylim(-0.05, 1.05)
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="best", fontsize=11, framealpha=0.92)

    plt.tight_layout()

    out_pdf = os.path.join(distilled_dir, args.output_name)
    base_name = os.path.splitext(args.output_name)[0]
    out_png = os.path.join(distilled_dir, f"{base_name}.png")

    save_figure(fig, out_pdf)
    save_figure(fig, out_png)
    plt.close(fig)

    print(f"Saved combined invariant sensitivity plot to:\n  - {out_pdf}\n  - {out_png}")


if __name__ == "__main__":
    main()
