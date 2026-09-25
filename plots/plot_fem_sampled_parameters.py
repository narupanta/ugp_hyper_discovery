#!/usr/bin/env python3
"""
plots/plot_fem_sampled_parameters.py

Plots the empirical posterior distributions, KDE densities, and joint parameter
scatter/correlations of the material parameters sampled and used for forward FEM
displacement field generation (from fem_distilled_samples.npz and distilled flow samples).
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import yaml

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "text.usetex": False
})


def parse_args():
    parser = argparse.ArgumentParser(description="Plot material parameters used in FEM validation.")
    parser.add_argument(
        "--model_path", type=str,
        default="results/20260910T085807_gentthomas_0.0001_0.01_2.0_0.95_5_0.01_isotropic_block/13/fem_validation",
        help="Path to fem_validation directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Custom output directory for plots (defaults to model_path and experiment plots/)"
    )
    return parser.parse_args()


def plot_fem_parameter_distributions(samples, true_dict, save_path):
    """
    Plots 1D KDE/Histogram marginals and pairwise joint distributions
    for the active parameters used in FEM forward simulation.
    """
    # Identify non-zero/active parameters
    active_indices = [i for i in range(samples.shape[1]) if np.std(samples[:, i]) > 1e-5 or np.abs(np.mean(samples[:, i])) > 1e-4]
    
    param_names = []
    param_labels = []
    for idx in active_indices:
        if idx < 10:
            param_names.append(f"dev_p{idx}")
            if idx == 0:
                param_labels.append(r"$C_{10}$ (Dev Param 1)")
            elif idx == 1:
                param_labels.append(r"$C_{01}$ (Dev Param 2)")
            else:
                param_labels.append(rf"$\theta_{{\mathrm{{dev}}, {idx}}}$")
        elif idx < 13:
            v_idx = idx - 10
            param_names.append(f"vol_p{v_idx}")
            if v_idx == 0:
                param_labels.append(r"$K$ (Bulk Modulus)")
            else:
                param_labels.append(rf"$\theta_{{\mathrm{{vol}}, {v_idx}}}$")
        else:
            a_idx = idx - 13
            param_names.append(f"aniso_p{a_idx}")
            param_labels.append(rf"$\theta_{{\mathrm{{aniso}}, {a_idx}}}$")

    n_params = len(active_indices)
    if n_params == 0:
        print("No active parameters found.")
        return

    # --- Figure 1: 1D Marginal Distributions ---
    fig, axes = plt.subplots(1, n_params, figsize=(5.2 * n_params, 4.5))
    if n_params == 1:
        axes = [axes]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (idx, name, lbl) in enumerate(zip(active_indices, param_names, param_labels)):
        ax = axes[i]
        vals = samples[:, idx]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        q025, q975 = np.percentile(vals, [2.5, 97.5])
        
        # KDE
        kde = gaussian_kde(vals)
        x_grid = np.linspace(min(vals) - 0.5 * std_val, max(vals) + 0.5 * std_val, 300)
        density = kde(x_grid)

        # Plot Histogram & KDE
        ax.hist(vals, bins=35, density=True, alpha=0.35, color=colors[i % len(colors)], edgecolor='black', lw=0.6)
        ax.plot(x_grid, density, color=colors[i % len(colors)], lw=2.5, label="Posterior Density")
        
        # Fill 95% CI
        mask_ci = (x_grid >= q025) & (x_grid <= q975)
        ax.fill_between(x_grid[mask_ci], density[mask_ci], color=colors[i % len(colors)], alpha=0.2, label=r"95% CI ($q_{0.025}, q_{0.975}$)")

        # Vertical lines
        ax.axvline(mean_val, color='navy', linestyle='-', lw=1.8, label=rf"Mean: {mean_val:.4f}")
        ax.axvline(q025, color=colors[i % len(colors)], linestyle=':', lw=1.5)
        ax.axvline(q975, color=colors[i % len(colors)], linestyle=':', lw=1.5)

        # True value if present
        true_val = true_dict.get(name, None)
        if true_val is not None:
            ax.axvline(true_val, color='red', linestyle='--', lw=2.0, label=rf"True: {true_val:.4f}")

        ax.set_title(lbl, fontsize=13, fontweight='bold')
        ax.set_xlabel("Parameter Value", fontsize=12)
        ax.set_ylabel("Probability Density", fontsize=12)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.legend(loc='best', fontsize=9.5, framealpha=0.9)

        # Text annotation with stats
        rel_std = (std_val / abs(mean_val) * 100.0) if abs(mean_val) > 1e-9 else 0.0
        stats_box = (
            rf"$\mu = {mean_val:.4f}$" + "\n"
            rf"$\sigma = {std_val:.4f} \ ({rel_std:.1f}\%)$" + "\n"
            rf"$95\%\ \mathrm{{CI}}: [{q025:.4f}, {q975:.4f}]$"
        )
        ax.text(0.04, 0.95, stats_box, transform=ax.transAxes, verticalalignment='top',
                fontsize=9.5, bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='#cccccc'))

    fig.suptitle(r"Material Parameter Uncertainty in FEM Forward Verification ($N = 1024$ Realizations)",
                 fontsize=15, y=1.02)
    plt.tight_layout()
    
    out_pdf = os.path.join(save_path, "fem_material_parameters_distribution.pdf")
    out_png = os.path.join(save_path, "fem_material_parameters_distribution.png")
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"✅ Saved 1D parameter distribution plots to:\n   {out_pdf}\n   {out_png}")

    # --- Figure 2: Pairwise Joint Distributions (Corner Plot) ---
    if n_params > 1:
        fig_pair, axes_pair = plt.subplots(n_params, n_params, figsize=(3.6 * n_params, 3.6 * n_params))
        for r in range(n_params):
            for c in range(n_params):
                ax = axes_pair[r, c]
                idx_r = active_indices[r]
                idx_c = active_indices[c]
                
                if r == c:
                    # Diagonal: 1D Histogram
                    ax.hist(samples[:, idx_r], bins=25, density=True, color=colors[r % len(colors)], alpha=0.6, edgecolor='black')
                    ax.grid(True, alpha=0.2)
                    if true_dict.get(param_names[r]) is not None:
                        ax.axvline(true_dict[param_names[r]], color='red', linestyle='--', lw=1.5)
                elif r > c:
                    # Off-diagonal: 2D Scatter + Contour
                    ax.scatter(samples[:, idx_c], samples[:, idx_r], s=12, alpha=0.35, color='#333333', edgecolors='none')
                    # True point
                    t_c = true_dict.get(param_names[c])
                    t_r = true_dict.get(param_names[r])
                    if t_c is not None and t_r is not None:
                        ax.scatter([t_c], [t_r], color='red', marker='*', s=120, zorder=5, label='True')
                    ax.grid(True, alpha=0.2)
                    # Correlation coefficient
                    corr = np.corrcoef(samples[:, idx_c], samples[:, idx_r])[0, 1]
                    ax.text(0.05, 0.90, rf"$\rho = {corr:.3f}$", transform=ax.transAxes, fontsize=10,
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='#ccc'))
                else:
                    ax.set_visible(False)

                if c == 0:
                    ax.set_ylabel(param_labels[r], fontsize=11)
                if r == n_params - 1:
                    ax.set_xlabel(param_labels[c], fontsize=11)

        fig_pair.suptitle(r"Joint Material Parameter Posterior Distributions", fontsize=15, y=0.99)
        plt.tight_layout()
        out_pair_pdf = os.path.join(save_path, "fem_material_parameters_joint.pdf")
        out_pair_png = os.path.join(save_path, "fem_material_parameters_joint.png")
        fig_pair.savefig(out_pair_pdf, bbox_inches='tight')
        fig_pair.savefig(out_pair_png, bbox_inches='tight', dpi=300)
        plt.close(fig_pair)
        print(f"✅ Saved joint parameter distribution plots to:\n   {out_pair_pdf}\n   {out_pair_png}")


def main():
    args = parse_args()
    
    # Try block or holes npz
    block_npz = os.path.join(args.model_path, "block", "fem_distilled_samples.npz")
    holes_npz = os.path.join(args.model_path, "holes", "fem_distilled_samples.npz")
    
    npz_path = block_npz if os.path.exists(block_npz) else holes_npz
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Could not find fem_distilled_samples.npz in {args.model_path}/block or /holes")

    data = np.load(npz_path)
    selected_samples = data["selected_samples"] # (N, 13)
    print(f"Loaded {len(selected_samples)} parameter samples with shape {selected_samples.shape} from {npz_path}")

    # Read config to get true parameters
    seed_dir = os.path.dirname(os.path.abspath(args.model_path))
    exp_dir = os.path.dirname(seed_dir)
    
    true_dict = {}
    for cfg_cand in [
        os.path.join(exp_dir, "config.yaml"),
        os.path.join(seed_dir, "extracted", "config.yaml"),
        os.path.join(seed_dir, "distilled", "config.yaml"),
        "configs/recipes/gentthomas.yaml"
    ]:
        if os.path.exists(cfg_cand):
            try:
                with open(cfg_cand, "r") as f:
                    cfg = yaml.safe_load(f)
                mat_p = cfg.get("material_params", {})
                dev_p = mat_p.get("dev_params", [])
                vol_p = mat_p.get("vol_params", [])
                for i, v in enumerate(dev_p):
                    true_dict[f"dev_p{i}"] = float(v)
                for i, v in enumerate(vol_p):
                    true_dict[f"vol_p{i}"] = float(v)
                print(f"Loaded true parameters from {cfg_cand}: {true_dict}")
                break
            except Exception as e:
                pass

    save_dirs = [args.model_path]
    top_plots = os.path.join(exp_dir, "plots")
    if os.path.exists(top_plots):
        save_dirs.append(top_plots)
    if args.output_dir:
        save_dirs.append(args.output_dir)

    for s_dir in set(save_dirs):
        os.makedirs(s_dir, exist_ok=True)
        plot_fem_parameter_distributions(selected_samples, true_dict, s_dir)

    # Check and plot failed FEM samples if present
    try:
        from plots.plot_failed_fem_samples import check_and_plot_failed_samples
        for s_dir in set(save_dirs):
            check_and_plot_failed_samples(s_dir)
            for sub in ["block", "holes"]:
                sub_p = os.path.join(s_dir, sub)
                if os.path.exists(sub_p):
                    check_and_plot_failed_samples(sub_p)
    except Exception as e:
        print(f"[WARN] Failed samples check encountered error: {e}")


if __name__ == "__main__":
    main()

