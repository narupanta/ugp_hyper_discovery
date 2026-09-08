"""
plots/plot_invariant_sensitivity_3d_pairs.py: 3D scatter plots of Sobol sensitivity indices vs invariant pairs.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plots.theme import apply_style, save_figure
from plots.sensitivity import load_sensitivity_data


def main():
    apply_style()
    parser = argparse.ArgumentParser(description="Plot invariant-dependent Sobol indices in 3D pairs")
    parser.add_argument("--distilled_dir", type=str, required=True, help="Path to distilled model directory")
    parser.add_argument("--active_params", type=str, default=None, help="Comma-separated list of active parameters")
    parser.add_argument("--component", type=str, default="dev", choices=["dev", "vol", "aniso"], help="Component for split model")
    parser.add_argument("--distill_target", type=str, default="sef", choices=["sef", "sef_stress", "sef_cauchy", "sef_split"])
    args = parser.parse_args()

    df, f3x3, invariants, params = load_sensitivity_data(
        args.distilled_dir,
        component=args.component,
        distill_target=args.distill_target,
        active_params=args.active_params
    )

    if df is None or invariants is None:
        print(f"Error: Unable to load sensitivity data or invariants for {args.distilled_dir}")
        return

    I1_bar, I2_bar, J = invariants
    I1_bar_m3 = I1_bar - 3.0
    I2_bar_m3 = I2_bar - 3.0
    J_m1_sq = (J - 1.0)**2

    if args.active_params:
        active_params = [p.strip() for p in args.active_params.split(',') if p.strip() in df.columns]
    else:
        active_params = [p for p in params if np.max(np.abs(df[p].values)) > 1e-4]

    if not active_params and params:
        active_params = [params[0]]

    if len(active_params) == 0:
        print("No active parameters found with sensitivity > 1e-4.")
        return

    fig = plt.figure(figsize=(18, 6))

    pairs = [
        (I1_bar_m3, r"$\bar{I}_1 - 3$", I2_bar_m3, r"$\bar{I}_2 - 3$"),
        (I1_bar_m3, r"$\bar{I}_1 - 3$", J_m1_sq, r"$(J - 1)^2$"),
        (I2_bar_m3, r"$\bar{I}_2 - 3$", J_m1_sq, r"$(J - 1)^2$")
    ]

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(active_params), 1)))
    axes = []

    for c_idx, (x_data, x_label, y_data, y_label) in enumerate(pairs):
        ax = fig.add_subplot(1, 3, c_idx + 1, projection='3d')
        axes.append(ax)

        for r_idx, param in enumerate(active_params):
            y_vals = df[param].values
            label_clean = param.replace("$", "")
            ax.scatter(x_data, y_data, y_vals, color=colors[r_idx], label=f"${label_clean}$", s=20, alpha=0.7)

        ax.set_title(f"Sensitivity vs {x_label} and {y_label}", fontsize=13)
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_zlabel(r"$S_T$", fontsize=11)
        ax.grid(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08),
                   ncol=len(active_params), fontsize=11, framealpha=0.9)

    plt.tight_layout()
    img_prefix = f"{args.component}_" if args.distill_target == "sef_split" else ""
    out_pdf = os.path.join(args.distilled_dir, f"{img_prefix}invariant_sensitivity_3d.pdf")
    out_png = os.path.join(args.distilled_dir, f"{img_prefix}invariant_sensitivity_3d.png")
    save_figure(fig, out_pdf)
    save_figure(fig, out_png)
    plt.close(fig)
    print(f"Plot saved to: {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
