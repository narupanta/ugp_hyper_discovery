"""
plots/plot_sobol_sensitivity.py: Bar plots of first-order and total-order Sobol indices with Pareto coverage.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plots.theme import apply_style, save_figure, COMPONENT_COLORS
from plots.sensitivity import get_sensitivity_statistics, find_sensitivity_dir


def main():
    apply_style()
    parser = argparse.ArgumentParser(description="Plot Sobol sensitivity indices")
    parser.add_argument("--distilled_dir", type=str, required=True, help="Path to distilled model directory")
    parser.add_argument("--sobol_threshold", type=float, default=0.001, help="Sobol threshold line")
    args = parser.parse_args()

    out_dir = args.distilled_dir
    is_split = (find_sensitivity_dir(out_dir, "dev_sensitivities") is not None)

    tot_means_all = {}
    first_means_all = {}
    param_types = {}

    if is_split:
        dev_tot, dev_first = get_sensitivity_statistics(out_dir, "dev_", "dev_sensitivities")
        vol_tot, vol_first = get_sensitivity_statistics(out_dir, "vol_", "vol_sensitivities")

        for k, v in dev_tot.items():
            tot_means_all[k] = v
            first_means_all[k] = dev_first.get(k, 0.0)
            param_types[k] = "dev"

        for k, v in vol_tot.items():
            tot_means_all[k] = v
            first_means_all[k] = vol_first.get(k, 0.0)
            param_types[k] = "vol"
    else:
        tot, first = get_sensitivity_statistics(out_dir, "", "sensitivities")
        for k, v in tot.items():
            tot_means_all[k] = v
            first_means_all[k] = first.get(k, 0.0)
            param_types[k] = "all"

    if not tot_means_all:
        print("No sensitivities found.")
        return

    sorted_params = sorted(tot_means_all.keys(), key=lambda k: tot_means_all[k], reverse=True)
    sorted_tot_means = np.array([tot_means_all[k] for k in sorted_params])
    sorted_first_means = np.array([first_means_all[k] for k in sorted_params])

    denominator = max(np.sum(sorted_first_means), np.sum(sorted_tot_means))
    if denominator == 0:
        denominator = 1.0

    est_coverage_frac = np.cumsum(sorted_tot_means) / denominator
    est_coverage_pct = est_coverage_frac * 100.0

    x = np.arange(len(sorted_params))
    width = 0.5

    fig, ax1 = plt.subplots(figsize=(12, 7))

    true_model_name = "isihara"
    parts = os.path.basename(os.path.normpath(args.distilled_dir)).split('_')
    for p in ["isihara", "nh", "neohookean2", "nh2", "gentthomas", "nh4", "neohookean4", "c20d10d05", "c20_d10_d05", "aniso30", "aniso_30"]:
        if p in parts:
            true_model_name = p
            break

    true_params = set()
    if true_model_name == "isihara":
        true_params = {"C10", "C01", "C20", "D1"}
    elif true_model_name in ["nh", "neohookean2", "nh2"]:
        true_params = {"C10", "D1"}
    elif true_model_name in ["nh4", "neohookean4"]:
        true_params = {"C10", "D2"}
    elif true_model_name in ["gentthomas"]:
        true_params = {"C10", "E", "CL2", "D1"}
    elif true_model_name in ["c20d10d05", "c20_d10_d05"]:
        true_params = {"C10", "D1", "D2"}
    elif true_model_name in ["aniso30", "aniso_30"]:
        true_params = {"C10", "D1", "K1", "K2", "aniso_theta"}

    added_gt_legend = False

    for i, p in enumerate(sorted_params):
        clean_p = p.replace("$", "").replace("{", "").replace("}", "").replace("_", "")

        if clean_p in true_params:
            if not added_gt_legend:
                ax1.axvspan(i - 0.5, i + 0.5, color='gray', alpha=0.15, zorder=0, label="Ground Truth Parameter")
                added_gt_legend = True
            else:
                ax1.axvspan(i - 0.5, i + 0.5, color='gray', alpha=0.15, zorder=0)

        ptype = param_types[p]
        if ptype == "dev":
            color = COMPONENT_COLORS["dev"]
            label = "Mean Total-Order (Dev)"
        elif ptype == "vol":
            color = COMPONENT_COLORS["vol"]
            label = "Mean Total-Order (Vol)"
        else:
            color = COMPONENT_COLORS["dev"]
            label = "Mean Total-Order"

        handles, labels = ax1.get_legend_handles_labels()
        if label not in labels:
            ax1.bar(x[i], sorted_tot_means[i], width, color=color, alpha=0.9, label=label)
        else:
            ax1.bar(x[i], sorted_tot_means[i], width, color=color, alpha=0.9)

    ax1.set_yscale('log')
    ax1.set_ylim(bottom=max(1e-5, args.sobol_threshold * 0.1))
    ax1.axhline(args.sobol_threshold, color='black', linestyle='--', linewidth=1.5, label=f"Threshold ({args.sobol_threshold})")

    ax1.set_ylabel('Sobol Sensitivity Index (Log Scale)', fontsize=12)
    ax1.set_title('Material Parameter Sensitivity (Sobol Indices)', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sorted_params, fontsize=11)
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, est_coverage_pct, color='black', marker='o', linestyle='-', linewidth=2, markersize=6, label="Estimated Coverage (EC)")
    ax2.set_ylabel('Estimated Coverage (%)', color='black', fontsize=12)
    ax2.set_ylim(0, 105)
    ax2.set_yticks([0, 20, 40, 60, 80, 95, 100])
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.axhline(100, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.axhline(95, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=11, loc='center right', bbox_to_anchor=(1.0, 0.6))

    plt.tight_layout()
    plot_name = "combined_sobol_sensitivity_indices.pdf" if is_split else "sobol_sensitivity_indices.pdf"
    sobol_plot_path = os.path.join(out_dir, plot_name)
    save_figure(fig, sobol_plot_path)
    plt.close(fig)
    print(f"Saved Sobol sensitivity plot to {sobol_plot_path}")


if __name__ == "__main__":
    main()
