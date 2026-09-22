"""
Plot the progression of parameter Sobol sensitivity across training load steps.
Concept 2: Line plots with mean curve and spatial min-max / percentile ribbons.
Can plot across the 4 training steps [2, 6, 10, 14], or all load steps.
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from plots.plot_domain_sensitivities import compute_element_sobol_sensitivities

def plot_sensitivity_progression(
    dataset_path="dataset/preprocessed/syn_f/isihara_1e-05_0.01_1.0_0.5_block_9.npz",
    distilled_dir="results/20260918T223851_isihara_1e-05_0.01_1.0_0.5_5_0.01_isotropic_block/9/distilled",
    steps=[2, 6, 10, 14],
    num_saltelli=256,
    save_dir=None
):
    if save_dir is None:
        save_dir = distilled_dir
    os.makedirs(save_dir, exist_ok=True)

    # 1. Load dataset
    data = np.load(dataset_path)
    if 'F_3d' in data:
        F_3d = data['F_3d']
    else:
        F_2d = data['F']
        F_3d = np.zeros((*F_2d.shape[:-2], 3, 3), dtype=np.float64)
        F_3d[..., :2, :2] = F_2d
        F_3d[..., 2, 2] = data.get('lam3', 1.0)

    # Load prior flow samples for bounds
    dev_samples_path = os.path.join(distilled_dir, "dev_flow_samples_before_sensitivity.npy")
    vol_samples_path = os.path.join(distilled_dir, "vol_flow_samples_before_sensitivity.npy")
    if not os.path.exists(dev_samples_path):
        dev_samples_path = os.path.join(distilled_dir, "dev_flow_samples.npy")
    if not os.path.exists(vol_samples_path):
        vol_samples_path = os.path.join(distilled_dir, "vol_flow_samples.npy")

    dev_samples = np.load(dev_samples_path)
    vol_samples = np.load(vol_samples_path)

    # 2. Compute sensitivities across requested steps
    print(f"Evaluating domain sensitivities across load steps {steps}...")
    step_st_dict = {}
    for s in steps:
        print(f"  Computing Step {s}...")
        st = compute_element_sobol_sensitivities(F_3d[s], dev_samples, vol_samples, num_saltelli=num_saltelli)
        step_st_dict[s] = st

    # Identify active parameters
    all_keys = list(step_st_dict[steps[-1]].keys())
    active_params = [k for k in all_keys if max(np.max(step_st_dict[s][k]) for s in steps) > 1e-3]
    # Rank by max sensitivity at final step
    active_params = sorted(active_params, key=lambda k: np.mean(step_st_dict[steps[-1]][k]), reverse=True)
    print(f"Active parameters plotted ({len(active_params)}): {active_params}")

    latex_names = {
        'C10': r'$C_{10}$', 'C01': r'$C_{01}$', 'C20': r'$C_{20}$',
        'C11': r'$C_{11}$', 'C02': r'$C_{02}$', 'C30': r'$C_{30}$',
        'C21': r'$C_{21}$', 'C12': r'$C_{12}$', 'C03': r'$C_{03}$', 'E': r'$E$',
        'D1': r'$D_1$', 'D2': r'$D_2$', 'D3': r'$D_3$'
    }

    # Consistent color palette
    colors = {
        'C10': '#1f77b4',
        'C01': '#4ba3e3',
        'C20': '#084594',
        'C11': '#009688',
        'C02': '#80cbc4',
        'D1': '#d95f02',
        'D2': '#fdae6b',
        'D3': '#7f2704'
    }

    # 3. Create Progression Figure
    # We will show two subplots:
    # Subplot 1: Deviatoric Parameters (C10, C01, C20, C11, C02)
    # Subplot 2: Volumetric Parameters (D1, D2)
    dev_active = [p for p in active_params if p.startswith('C') or p == 'E']
    vol_active = [p for p in active_params if p.startswith('D')]

    fig, (ax_dev, ax_vol) = plt.subplots(1, 2, figsize=(8.0, 3.2), sharex=True)
    plt.subplots_adjust(wspace=0.28, top=0.90, bottom=0.15, left=0.08, right=0.96)

    x_vals = np.array(steps)

    # Plot Deviatoric terms
    for p in dev_active:
        means = [np.mean(step_st_dict[s][p]) for s in steps]
        mins = [np.min(step_st_dict[s][p]) for s in steps]
        maxs = [np.max(step_st_dict[s][p]) for s in steps]
        col = colors.get(p, '#333333')

        ax_dev.plot(x_vals, means, marker='o', markersize=4.5, linewidth=1.8, color=col, label=latex_names.get(p, p))
        ax_dev.fill_between(x_vals, mins, maxs, color=col, alpha=0.15)

    ax_dev.set_title("Deviatoric Parameters Sensitivity", fontsize=10.0, fontweight='bold', pad=6)
    ax_dev.set_xlabel("Load Step", fontsize=9.0, labelpad=2)
    ax_dev.set_ylabel(r"Total Sobol Sensitivity $S_{\mathrm{T,d}}$", fontsize=9.0, labelpad=2)
    ax_dev.set_ylim(-0.02, 1.02)
    ax_dev.set_xticks(steps)
    ax_dev.set_xticklabels([f"Step {s}" for s in steps], fontsize=8.5)
    ax_dev.tick_params(axis='both', labelsize=8.0)
    ax_dev.grid(True, linestyle=':', alpha=0.5)
    ax_dev.legend(fontsize=8.0, frameon=True, facecolor='white', framealpha=0.9, edgecolor='#dddddd', loc='center left')

    # Plot Volumetric terms
    for p in vol_active:
        means = [np.mean(step_st_dict[s][p]) for s in steps]
        mins = [np.min(step_st_dict[s][p]) for s in steps]
        maxs = [np.max(step_st_dict[s][p]) for s in steps]
        col = colors.get(p, '#333333')

        ax_vol.plot(x_vals, means, marker='s', markersize=4.5, linewidth=1.8, color=col, label=latex_names.get(p, p))
        ax_vol.fill_between(x_vals, mins, maxs, color=col, alpha=0.15)

    ax_vol.set_title("Volumetric Parameters Sensitivity", fontsize=10.0, fontweight='bold', pad=6)
    ax_vol.set_xlabel("Load Step", fontsize=9.0, labelpad=2)
    ax_vol.set_ylabel(r"Total Sobol Sensitivity $S_{\mathrm{T,v}}$", fontsize=9.0, labelpad=2)
    ax_vol.set_ylim(-0.02, 1.02)
    ax_vol.set_xticks(steps)
    ax_vol.set_xticklabels([f"Step {s}" for s in steps], fontsize=8.5)
    ax_vol.tick_params(axis='both', labelsize=8.0)
    ax_vol.grid(True, linestyle=':', alpha=0.5)
    ax_vol.legend(fontsize=8.0, frameon=True, facecolor='white', framealpha=0.9, edgecolor='#dddddd', loc='center right')

    out_pdf = os.path.join(save_dir, "sensitivity_progression_across_steps.pdf")
    out_png = os.path.join(save_dir, "sensitivity_progression_across_steps.png")
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved sensitivity progression figure to:\n  - {out_pdf}\n  - {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="dataset/preprocessed/syn_f/isihara_1e-05_0.01_1.0_0.5_block_9.npz")
    parser.add_argument("--distilled_dir", type=str, default="results/20260918T223851_isihara_1e-05_0.01_1.0_0.5_5_0.01_isotropic_block/9/distilled")
    parser.add_argument("--steps", nargs="+", type=int, default=[2, 6, 10, 14])
    args = parser.parse_args()

    plot_sensitivity_progression(
        dataset_path=args.dataset_path,
        distilled_dir=args.distilled_dir,
        steps=args.steps
    )
