"""
Plot domain sensitivity maps where the color of each discrete finite element
represents the Sobol sensitivity index of a specific distilled parameter.
Supports standalone generation across load steps.
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

def compute_element_sobol_sensitivities(F_3d_step, dev_samples, vol_samples, num_saltelli=256):
    """
    Computes Sobol total-order sensitivity indices S_T for each element given its 3D deformation gradient.
    F_3d_step: (num_elements, 3, 3)
    dev_samples: (N_samples, 10)
    vol_samples: (N_samples, 3)
    """
    num_elements = F_3d_step.shape[0]
    dev_names = ['C10', 'C01', 'C20', 'C11', 'C02', 'C30', 'C21', 'C12', 'C03', 'E']
    vol_names = ['D1', 'D2', 'D3']
    
    # Invariant calculations per element
    # C = F^T F
    C = np.einsum('nij,nik->njk', F_3d_step, F_3d_step)
    I1 = np.trace(C, axis1=1, axis2=2)
    C2 = np.einsum('nij,njk->nik', C, C)
    I2 = 0.5 * (I1**2 - np.trace(C2, axis1=1, axis2=2))
    J = np.linalg.det(F_3d_step)
    
    J_safe = np.clip(J, 1e-8, 1e8)
    I1_bar = (J_safe ** (-2.0 / 3.0)) * I1
    I2_bar = (J_safe ** (-4.0 / 3.0)) * I2
    
    i1_m3 = I1_bar - 3.0
    i2_m3 = I2_bar - 3.0
    J_m1 = J - 1.0
    
    # 1. Deviatoric Sobol Analysis
    dev_bounds = np.column_stack([dev_samples.min(axis=0), dev_samples.max(axis=0)])
    # Protect against zero range for inactive terms
    for idx in range(dev_bounds.shape[0]):
        if dev_bounds[idx, 1] - dev_bounds[idx, 0] < 1e-8:
            dev_bounds[idx, 1] = dev_bounds[idx, 0] + 1e-6
            
    dev_problem = {
        'num_vars': len(dev_names),
        'names': dev_names,
        'bounds': dev_bounds.tolist()
    }
    dev_eval_params = sobol_sample.sample(dev_problem, num_saltelli, calc_second_order=False) # (N_eval, 10)
    
    # Vectorized dev terms: (10, num_elements)
    dev_terms = np.stack([
        i1_m3,
        i2_m3,
        i1_m3**2,
        i1_m3 * i2_m3,
        i2_m3**2,
        i1_m3**3,
        (i1_m3**2) * i2_m3,
        i1_m3 * (i2_m3**2),
        i2_m3**3,
        np.log(np.maximum(I2_bar / 3.0, 1e-8))
    ], axis=0)
    
    # Evaluate dev energy: (N_eval, num_elements)
    W_dev = np.dot(dev_eval_params, dev_terms)
    
    ST_dev = np.zeros((num_elements, len(dev_names)))
    for el in range(num_elements):
        # Check variance of output
        if np.var(W_dev[:, el]) < 1e-12:
            ST_dev[el, :] = 0.0
        else:
            res = sobol_analyze.analyze(dev_problem, W_dev[:, el], calc_second_order=False, print_to_console=False)
            ST_dev[el, :] = np.clip(np.nan_to_num(res['ST'], nan=0.0), 0.0, 1.0)
            
    # 2. Volumetric Sobol Analysis
    vol_bounds = np.column_stack([vol_samples.min(axis=0), vol_samples.max(axis=0)])
    for idx in range(vol_bounds.shape[0]):
        if vol_bounds[idx, 1] - vol_bounds[idx, 0] < 1e-8:
            vol_bounds[idx, 1] = vol_bounds[idx, 0] + 1e-6
            
    vol_problem = {
        'num_vars': len(vol_names),
        'names': vol_names,
        'bounds': vol_bounds.tolist()
    }
    vol_eval_params = sobol_sample.sample(vol_problem, num_saltelli, calc_second_order=False) # (N_eval, 3)
    
    vol_terms = np.stack([
        J_m1**2,
        J_m1**4,
        J_m1**6
    ], axis=0) # (3, num_elements)
    
    W_vol = np.dot(vol_eval_params, vol_terms) # (N_eval, num_elements)
    
    ST_vol = np.zeros((num_elements, len(vol_names)))
    for el in range(num_elements):
        if np.var(W_vol[:, el]) < 1e-12:
            ST_vol[el, :] = 0.0
        else:
            res = sobol_analyze.analyze(vol_problem, W_vol[:, el], calc_second_order=False, print_to_console=False)
            ST_vol[el, :] = np.clip(np.nan_to_num(res['ST'], nan=0.0), 0.0, 1.0)
            
    # Combine results into dictionary
    st_dict = {}
    for idx, name in enumerate(dev_names):
        st_dict[name] = ST_dev[:, idx]
    for idx, name in enumerate(vol_names):
        st_dict[name] = ST_vol[:, idx]
        
    return st_dict


def plot_domain_sensitivities(dataset_path, distilled_dir, step_idx=14, num_saltelli=256, save_dir=None):
    """
    Renders domain sensitivity plots for all distilled parameters.
    """
    if save_dir is None:
        save_dir = distilled_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Load dataset mesh and deformation gradient
    data = np.load(dataset_path)
    mesh_pos = data['mesh_pos']
    cells = data['cells']
    
    if 'F_3d' in data:
        F_3d_all = data['F_3d']
    else:
        # Fallback build 3D from F and lam3
        F_2d = data['F']
        F_3d_all = np.zeros((*F_2d.shape[:-2], 3, 3), dtype=np.float64)
        F_3d_all[..., :2, :2] = F_2d
        if 'lam3' in data:
            F_3d_all[..., 2, 2] = data['lam3']
        else:
            F_3d_all[..., 2, 2] = 1.0
            
    num_steps = F_3d_all.shape[0]
    if step_idx < 0:
        step_idx = num_steps + step_idx
    step_idx = max(0, min(step_idx, num_steps - 1))
    
    F_3d_step = F_3d_all[step_idx] # (528, 3, 3)
    
    # Load displaced mesh positions if available for deformed plot, or undeformed mesh
    if 'u_true' in data or 'u' in data:
        u_arr = data.get('u_true', data.get('u'))
        u_step = u_arr[step_idx]
        plot_mesh = mesh_pos + u_step
    else:
        plot_mesh = mesh_pos

    triangulation = tri.Triangulation(plot_mesh[:, 0], plot_mesh[:, 1], cells)
    
    # 2. Load flow samples before sensitivity to get prior parameter bounds
    dev_samples_path = os.path.join(distilled_dir, "dev_flow_samples_before_sensitivity.npy")
    vol_samples_path = os.path.join(distilled_dir, "vol_flow_samples_before_sensitivity.npy")
    if not os.path.exists(dev_samples_path):
        dev_samples_path = os.path.join(distilled_dir, "dev_flow_samples.npy")
    if not os.path.exists(vol_samples_path):
        vol_samples_path = os.path.join(distilled_dir, "vol_flow_samples.npy")
        
    dev_samples = np.load(dev_samples_path)
    vol_samples = np.load(vol_samples_path)
    
    # 3. Compute exact element-wise Sobol sensitivities
    print(f"Computing element-wise Sobol sensitivities for {len(cells)} triangles at Step {step_idx}...")
    st_dict = compute_element_sobol_sensitivities(F_3d_step, dev_samples, vol_samples, num_saltelli=num_saltelli)
    
    # 4. Filter to active distilled parameters (e.g. 7 parameters: D1, C10, C01, D2, C20, C11, C02)
    # Ranked by their mean sensitivity in the domain
    active_params = [k for k, v in st_dict.items() if np.max(v) > 1e-4]
    # Sort active params by descending mean sensitivity
    active_params = sorted(active_params, key=lambda k: np.mean(st_dict[k]), reverse=True)
    print(f"Active distilled parameters with non-zero sensitivity ({len(active_params)}): {active_params}")
    
    n_params = len(active_params)
    if n_params == 0:
        print("No active parameters found!")
        return
        
    # Standard formatting names
    latex_names = {
        'C10': r'$C_{10}$', 'C01': r'$C_{01}$', 'C20': r'$C_{20}$',
        'C11': r'$C_{11}$', 'C02': r'$C_{02}$', 'C30': r'$C_{30}$',
        'C21': r'$C_{21}$', 'C12': r'$C_{12}$', 'C03': r'$C_{03}$', 'E': r'$E$',
        'D1': r'$D_1$', 'D2': r'$D_2$', 'D3': r'$D_3$'
    }
    
    # Layout for 7 parameters: 2 rows (4 on top, 3 on bottom, or 1 row of 7, or 2x4 with 1 empty)
    ncols = min(4, n_params)
    nrows = int(np.ceil(n_params / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), constrained_layout=True)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[None, :]
    elif ncols == 1:
        axes = axes[:, None]
        
    for i, p_name in enumerate(active_params):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        val = st_dict[p_name]
        
        # Determine local range to maximize perceptual contrast across spatial gradients
        # Use robust percentiles to prevent single element outliers while maximizing domain gradient visibility
        v_min_local = max(0.0, float(np.min(val)))
        v_max_local = min(1.0, float(np.max(val)))
        if v_max_local - v_min_local < 0.05:
            # Expand slightly around mean if almost constant
            v_mid = float(np.mean(val))
            v_min_plot = max(0.0, v_mid - 0.05)
            v_max_plot = min(1.0, v_mid + 0.05)
        else:
            v_min_plot = v_min_local
            v_max_plot = v_max_local

        # High-contrast 'plasma' or 'inferno' colormap provides far better dynamic contrast than muted viridis
        tpc = ax.tripcolor(triangulation, facecolors=val, cmap='plasma', vmin=v_min_plot, vmax=v_max_plot,
                           edgecolors='black', linewidth=0.15)
        ax.set_aspect('equal')
        ax.set_title(f"{latex_names.get(p_name, p_name)} ($S_T$)\n[{v_min_local:.2f}, {v_max_local:.2f}]", fontsize=10.5, fontweight='bold', pad=4)
        ax.set_xlabel("X [mm]", fontsize=8.5, labelpad=2)
        ax.set_ylabel("Y [mm]", fontsize=8.5, labelpad=2)
        ax.tick_params(labelsize=7.5)
        
        cbar = fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$S_T$", fontsize=8.5)
        cbar.ax.tick_params(labelsize=7.5)
        
    # Hide unused subplots if any
    for j in range(n_params, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].set_visible(False)
        
    fig.suptitle(f"Parameter Sobol Sensitivity Distributions Across Physical Domain (Step {step_idx})", fontsize=13, fontweight='bold')
    
    out_pdf = os.path.join(save_dir, f"domain_parameter_sensitivities_step{step_idx}.pdf")
    out_png = os.path.join(save_dir, f"domain_parameter_sensitivities_step{step_idx}.png")
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Successfully saved domain sensitivity plots to:\n  - {out_pdf}\n  - {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot domain sensitivity maps for distilled parameters.")
    parser.add_argument("--dataset_path", type=str, default="dataset/preprocessed/syn_f/isihara_1e-05_0.01_1.0_0.5_block_9.npz")
    parser.add_argument("--distilled_dir", type=str, default="results/20260918T223851_isihara_1e-05_0.01_1.0_0.5_5_0.01_isotropic_block/9/distilled")
    parser.add_argument("--step_idx", type=int, default=14, help="Load step to visualize (e.g. 14 for end of training, 19 for extrapolation)")
    parser.add_argument("--num_saltelli", type=int, default=256, help="Saltelli sample multiplier per parameter")
    args = parser.parse_args()
    
    plot_domain_sensitivities(
        dataset_path=args.dataset_path,
        distilled_dir=args.distilled_dir,
        step_idx=args.step_idx,
        num_saltelli=args.num_saltelli
    )

