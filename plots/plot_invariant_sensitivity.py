"""
plots/plot_invariant_sensitivity.py: Scatter plots of Sobol indices vs strain invariants.
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
    parser = argparse.ArgumentParser(description="Plot invariant-dependent Sobol indices")
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

    if args.active_params:
        active_params = [p.strip() for p in args.active_params.split(',') if p.strip() in df.columns]
    else:
        active_params = [p for p in params if np.max(np.abs(df[p].values)) > 1e-4]

    if not active_params and params:
        active_params = [params[0]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    inv_list = [
        (r"$\bar{I}_1$", I1_bar),
        (r"$\bar{I}_2$", I2_bar),
        (r"$J$", J)
    ]

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(active_params), 1)))

    for i, (inv_name, inv_vals) in enumerate(inv_list):
        ax = axes[i]
        for j, param in enumerate(active_params):
            y_vals = df[param].values
            label_clean = param.replace("$", "")
            ax.scatter(inv_vals, y_vals, label=f"${label_clean}$", alpha=0.75, s=24, color=colors[j])

        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Sensitivity vs {inv_name}", fontsize=13)
        ax.set_xlabel(inv_name, fontsize=12)
        ax.set_ylabel(r"Total-Order Sobol Index ($S_T$)", fontsize=12)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08),
                   ncol=len(active_params), fontsize=11, framealpha=0.9)

    plt.tight_layout()
    img_prefix = f"{args.component}_" if args.distill_target == "sef_split" else ""
    out_pdf = os.path.join(args.distilled_dir, f"{img_prefix}invariant_sensitivity_scatter.pdf")
    out_png = os.path.join(args.distilled_dir, f"{img_prefix}invariant_sensitivity_scatter.png")
    save_figure(fig, out_pdf)
    save_figure(fig, out_png)
    plt.close(fig)
    print(f"Plot saved to: {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
