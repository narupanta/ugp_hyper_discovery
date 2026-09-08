"""
plots/plot_deformation_sensitivity.py: Plots total-order Sobol indices across the 6 deformation modes.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plots.theme import apply_style, save_figure, MODE_NAMES
from plots.sensitivity import load_sobol_csv, find_sensitivity_dir
from core.utils import load_f3x3_from_distilled


def extract_gamma(F, mode_idx):
    """Extract stretch measure gamma from deformation gradient F based on mode index."""
    if mode_idx in [0, 1, 2]:  # UT, EBT, PS
        return F[0, 0] - 1.0
    elif mode_idx in [3, 4]:  # UC, EBC
        return 1.0 / F[0, 0] - 1.0
    elif mode_idx == 5:  # SS
        return F[0, 1]
    return 0.0


def main():
    apply_style()
    parser = argparse.ArgumentParser(description="Plot deformation-dependent Sobol indices")
    parser.add_argument("--distilled_dir", type=str, required=True, help="Path to distilled model directory")
    parser.add_argument("--active_params", type=str, default=None, help="Comma-separated list of active parameters")
    parser.add_argument("--component", type=str, default="dev", choices=["dev", "vol", "aniso"], help="Component for split model")
    parser.add_argument("--distill_target", type=str, default="sef", choices=["sef", "sef_stress", "sef_cauchy", "sef_split"])
    args = parser.parse_args()

    subdir = f"{args.component}_sensitivities" if args.distill_target == "sef_split" else "sensitivities"
    img_prefix = f"{args.component}_" if args.distill_target == "sef_split" else ""

    df, csv_path = load_sobol_csv(args.distilled_dir, subdir, "total_sobol_indices_output_0.csv")
    if df is None:
        print(f"Error: Sobol CSV not found for {args.distilled_dir} in {subdir}")
        return

    exclude_cols = ['test cases', 'Unnamed: 0']
    params = [c for c in df.columns if c not in exclude_cols and not c.startswith('Unnamed')]

    if args.active_params:
        active_list = [p.strip() for p in args.active_params.split(',') if p.strip() in params]
        if active_list:
            params = active_list

    f3x3 = load_f3x3_from_distilled(args.distilled_dir)
    if f3x3 is None:
        print(f"Error: f3x3.npy not found for {args.distilled_dir}")
        return

    num_orig_points = f3x3.shape[0]
    keep_mask = np.ones(num_orig_points, dtype=bool)
    identity_matrix = np.eye(3)
    is_zero_strain = np.max(np.max(np.abs(f3x3 - identity_matrix), axis=2), axis=1) < 1e-6
    keep_mask[is_zero_strain] = False

    f3x3_kept = f3x3[keep_mask]
    if len(f3x3_kept) != len(df):
        min_len = min(len(f3x3_kept), len(df))
        f3x3_kept = f3x3_kept[:min_len]
        df = df.iloc[:min_len]

    chunk_size = max(1, num_orig_points // 6)
    orig_indices_kept = np.arange(num_orig_points)[keep_mask][:len(df)]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for mode_idx in range(6):
        row_mask = (orig_indices_kept // chunk_size) == mode_idx
        df_mode = df[row_mask]
        f3x3_mode = f3x3_kept[row_mask]

        if len(df_mode) == 0:
            continue

        gammas = [extract_gamma(F, mode_idx) for F in f3x3_mode]
        ax = axes[mode_idx]

        plotted = 0
        for param in params:
            y_vals = df_mode[param].values
            label_clean = param.replace("$", "")
            if np.max(np.abs(y_vals)) > 1e-4:
                ax.plot(gammas, y_vals, label=f"${label_clean}$", linewidth=2.0, alpha=0.85, marker='x', markersize=5)
                plotted += 1

        if plotted == 0:
            for param in params:
                label_clean = param.replace("$", "")
                ax.plot(gammas, df_mode[param].values, label=f"${label_clean}$", linewidth=1.2, alpha=0.5, marker='x', markersize=4)

        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"({mode_idx + 1}) {MODE_NAMES[mode_idx]}", fontsize=13)
        ax.set_xlabel(r"Stretch Measure ($\gamma$)", fontsize=11)
        if mode_idx % 3 == 0:
            ax.set_ylabel(r"Total-Order Sobol Index ($S_T$)", fontsize=11)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
                   ncol=max(1, len(params) // 2 + 1), fontsize=11, framealpha=0.9)

    plt.tight_layout()
    plot_pdf = os.path.join(args.distilled_dir, f"{img_prefix}deformation_sensitivity_modes.pdf")
    plot_png = os.path.join(args.distilled_dir, f"{img_prefix}deformation_sensitivity_modes.png")
    save_figure(fig, plot_pdf)
    save_figure(fig, plot_png)
    plt.close(fig)
    print(f"Deformation sensitivity plot saved to {plot_pdf} and {plot_png}")


if __name__ == "__main__":
    main()
