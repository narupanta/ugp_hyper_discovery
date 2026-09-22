"""
Plot invariant sensitivities across 4 training load steps (Steps 2, 6, 10, 14).
Generates a 4-row x 3-column figure:
  Columns:
    1: S_T,d vs I1_bar
    2: S_T,d vs I2_bar
    3: S_T,v vs J
  Rows:
    Row 1: Step 2
    Row 2: Step 6
    Row 3: Step 10
    Row 4: Step 14
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from plots.plot_domain_sensitivities import compute_element_sobol_sensitivities

def to_latex(name):
    clean = name.replace('$', '').replace('{', '').replace('}', '').replace('_', '')
    if clean.startswith('C'):
        return f"$C_{{{clean[1:]}}}$"
    elif clean.startswith('D'):
        return f"$D_{{{clean[1:]}}}$"
    return name

def get_comp_color(name):
    palette = {
        'C10': '#0072B2',
        'C01': '#56B4E9',
        'C20': '#084594',
        'C11': '#009E73',
        'C02': '#80cbc4',
        'C30': '#CC79A7',
        'C21': '#D55E00',
        'C12': '#E69F00',
        'C03': '#F0E442',
        'E':   '#999999',
        'D1':  '#D55E00',
        'D2':  '#F0E442',
        'D3':  '#7f2704'
    }
    clean = name.replace('$', '').replace('{', '').replace('}', '').replace('_', '')
    return palette.get(clean, '#333333')

def plot_multistep_invariant_sensitivities(
    dataset_path="dataset/preprocessed/syn_f/isihara_1e-05_0.01_1.0_0.5_block_9.npz",
    distilled_dir="results/20260918T223851_isihara_1e-05_0.01_1.0_0.5_5_0.01_isotropic_block/9/distilled",
    steps=[2, 6, 10, 14],
    num_saltelli=256,
    save_dir=None
):
    if save_dir is None:
        save_dir = distilled_dir
    os.makedirs(save_dir, exist_ok=True)

    # 1. Load data
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

    # 2. Compute sensitivities & invariants for each step
    step_data = {}
    print(f"Computing sensitivities for steps {steps}...")
    for s in steps:
        F_step = F_3d[s]
        # Invariants:
        C = np.einsum('nij,nik->njk', F_step, F_step)
        I1 = np.trace(C, axis1=1, axis2=2)
        C2 = np.einsum('nij,njk->nik', C, C)
        I2 = 0.5 * (I1**2 - np.trace(C2, axis1=1, axis2=2))
        J = np.linalg.det(F_step)
        J_safe = np.clip(J, 1e-8, 1e8)
        I1_bar = (J_safe ** (-2.0 / 3.0)) * I1
        I2_bar = (J_safe ** (-4.0 / 3.0)) * I2

        st = compute_element_sobol_sensitivities(F_step, dev_samples, vol_samples, num_saltelli=num_saltelli)
        step_data[s] = {
            'I1_bar': I1_bar,
            'I2_bar': I2_bar,
            'J': J,
            'st': st
        }

    # Find overall active parameters across all 4 steps
    all_keys = list(step_data[steps[-1]]['st'].keys())
    active_params = [k for k in all_keys if max(np.max(step_data[s]['st'][k]) for s in steps) > 1e-3]
    active_dev = [k for k in active_params if k.startswith('C') or k == 'E']
    active_vol = [k for k in active_params if k.startswith('D')]
    
    print(f"Active dev params: {active_dev}")
    print(f"Active vol params: {active_vol}")

    # Determine global x-limits across all steps for shared horizontal axes
    all_I1 = np.concatenate([step_data[s]['I1_bar'] for s in steps])
    all_I2 = np.concatenate([step_data[s]['I2_bar'] for s in steps])
    all_J = np.concatenate([step_data[s]['J'] for s in steps])

    xlim_I1 = (2.95, np.max(all_I1) + 0.05)
    xlim_I2 = (2.95, np.max(all_I2) + 0.05)
    xlim_J = (np.min(all_J) - 0.05, np.max(all_J) + 0.05)

    # 3. Create 4x3 Figure
    nrows = len(steps)
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.8, 1.9 * nrows), constrained_layout=True)

    for row_idx, s in enumerate(steps):
        ax_i1 = axes[row_idx, 0]
        ax_i2 = axes[row_idx, 1]
        ax_j  = axes[row_idx, 2]

        I1_b = step_data[s]['I1_bar']
        I2_b = step_data[s]['I2_bar']
        J_val = step_data[s]['J']
        st_s = step_data[s]['st']

        # Formatting each subplot
        for ax in [ax_i1, ax_i2, ax_j]:
            ax.set_box_aspect(1)
            ax.grid(False)
            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.tick_params(axis='both', which='major', labelsize=6.8, pad=1)

        ax_i1.set_xlim(xlim_I1)
        ax_i2.set_xlim(xlim_I2)
        ax_j.set_xlim(xlim_J)

        # Plot Dev: ST,d vs I1_bar
        for p in active_dev:
            col = get_comp_color(p)
            ax_i1.scatter(I1_b, st_s[p], color=col, alpha=0.65, s=9, edgecolors='none')
            ax_i2.scatter(I2_b, st_s[p], color=col, alpha=0.65, s=9, edgecolors='none')

        # Plot Vol: ST,v vs J
        for p in active_vol:
            col = get_comp_color(p)
            ax_j.scatter(J_val, st_s[p], color=col, alpha=0.65, s=9, edgecolors='none')

        # Labels
        ax_i1.set_ylabel(r"$S_{\mathrm{T,d}}$", fontsize=8.0, labelpad=2)
        ax_i2.set_ylabel(r"$S_{\mathrm{T,d}}$", fontsize=8.0, labelpad=2)
        ax_j.set_ylabel(r"$S_{\mathrm{T,v}}$", fontsize=8.0, labelpad=2)

        # Step annotation on the left
        ax_i1.text(-0.35, 0.5, f"Step {s}", transform=ax_i1.transAxes,
                   rotation=90, va='center', ha='center', fontsize=9.5, fontweight='bold')

        if row_idx == nrows - 1:
            ax_i1.set_xlabel(r"$\bar{I}_1$", fontsize=8.5, labelpad=2)
            ax_i2.set_xlabel(r"$\bar{I}_2$", fontsize=8.5, labelpad=2)
            ax_j.set_xlabel(r"$J$", fontsize=8.5, labelpad=2)
        else:
            ax_i1.set_xticklabels([])
            ax_i2.set_xticklabels([])
            ax_j.set_xticklabels([])

    # Unified bottom legend
    all_active = list(dict.fromkeys(active_dev + active_vol))
    inv_handles = []
    inv_labels = []
    for p in all_active:
        col = get_comp_color(p)
        inv_handles.append(mlines.Line2D([], [], color=col, marker='o', linestyle='none', markersize=5))
        inv_labels.append(to_latex(p))

    fig.legend(handles=inv_handles, labels=inv_labels, loc='lower center',
               bbox_to_anchor=(0.5, -0.04), ncol=len(all_active),
               fontsize=7.8, frameon=False, handletextpad=0.2, columnspacing=1.0)

    out_pdf = os.path.join(save_dir, "invariant_sensitivities_multistep.pdf")
    out_png = os.path.join(save_dir, "invariant_sensitivities_multistep.png")
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved multistep invariant sensitivities to:\n  - {out_pdf}\n  - {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="dataset/preprocessed/syn_f/isihara_1e-05_0.01_1.0_0.5_block_9.npz")
    parser.add_argument("--distilled_dir", type=str, default="results/20260918T223851_isihara_1e-05_0.01_1.0_0.5_5_0.01_isotropic_block/9/distilled")
    parser.add_argument("--steps", nargs="+", type=int, default=[2, 6, 10, 14])
    args = parser.parse_args()

    plot_multistep_invariant_sensitivities(
        dataset_path=args.dataset_path,
        distilled_dir=args.distilled_dir,
        steps=args.steps
    )

