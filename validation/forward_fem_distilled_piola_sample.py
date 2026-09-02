
# Import some useful modules.
import jax
import jax.numpy as jnp
import os
from pathlib import Path
# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh
import jax.random as jr 
jax.config.update("jax_enable_x64", True)

from core.utils import *
from core.model import SparseHyperelasticityGP
from core.dataclass import GPParams, GPRawParams
from core.material_models import get_material
from core.datasetclass import BenchmarkDataset

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import LinearSegmentedColormap

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import r2_score
import argparse
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"], # Falls back to DejaVu if Times isn't found
    "font.size": 16,                # Base font size
    "axes.titlesize": 18,           # Subplot titles
    "axes.labelsize": 14,           # X and Y labels
    "legend.fontsize": 12,          # Legend text
    "xtick.labelsize": 12,          # Axis tick numbers
    "ytick.labelsize": 12,
    "figure.dpi": 600,              # High resolution for the screen and save
    "savefig.dpi": 600,             # Ensures saved file is high quality
    "text.usetex": False            # Set to True only if you have a full LaTeX install
})


def plot_fem_verification(I1_bar_true, I2_bar_true, J_true,
                          I1_bar_pred, I2_bar_pred, J_pred,
                          I1_bar_train, I2_bar_train, J_train,
                          inducing_points, save_path):

    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))

    # Data groups for iteration
    data = [
        (I1_bar_true, I1_bar_pred, r'$\bar{I}_1$', axes1[0]),
        (I2_bar_true, I2_bar_pred, r'$\bar{I}_2$', axes1[1]),
        (J_true, J_pred, r'$J$', axes1[2])
    ]

    for true, pred, name, ax in data:
        # Calculate R2
        r2 = r2_score(true.flatten(), pred.flatten())
        
        # Plot Scatter (Hollow circles)
        ax.scatter(true, pred, alpha=0.5, edgecolors='b', facecolors='none', s=20)
        
        # Identity Line (Perfect parity)
        all_vals = np.concatenate([true.flatten(), pred.flatten()])
        low, high = np.min(all_vals), np.max(all_vals)
        ax.plot([low, high], [low, high], 'k--', alpha=0.75, zorder=0)
        
        # Add R2 Text Box
        ax.text(0.05, 0.92, f'$R^2 = {r2:.4f}$', transform=ax.transAxes, 
                fontsize=14, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Formatting
        ax.set_title(f'{name} Accuracy', fontsize = 16)
        ax.set_xlabel('True')
        if ax == axes1[0]:
            ax.set_ylabel('Pred')
        
        ax.grid(False) # Consistent with your previous request
        ax.set_xlim([low, high])
        ax.set_ylim([low, high])
    fig1.suptitle("Predicted Material Accuracy on FEM", fontsize=20)
    fig1.tight_layout()
    
    # Save files
    os.makedirs(save_path, exist_ok=True)
    fig1.savefig(os.path.join(save_path, "fem_deployment_accuracy_r2.pdf"), dpi=600)
    # --- FIGURE 2: Invariant Space Coverage ---
    # Inducing points indices: 0:I1_bar, 1:I2_bar, 2:J
    # z_i1 = inducing_points[:, 0]
    # z_i2 = inducing_points[:, 1]
    # z_j  = inducing_points[:, 2]

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

    # I1_bar vs I2_bar
    axes2[0].scatter(I1_bar_pred - 3, I2_bar_pred- 3, edgecolors='red', facecolors='none', alpha=0.3, label='Pred')
    axes2[0].scatter(I1_bar_train- 3, I2_bar_train- 3, edgecolors='black', facecolors='none', alpha=0.5, label='Train')
    # axes2[0].scatter(z_i1- 3, z_i2 - 3, color='blue', marker='x', s=50, label='Inducing')
    axes2[0].set_xlabel(r'$\bar{I}_1 - 3$')
    axes2[0].set_ylabel(r'$\bar{I}_2 - 3$')
    axes2[0].set_title(r'$\bar{I}_1 - 3$ vs $\bar{I}_2 - 3$')
    axes2[0].legend()

    # I1_bar vs J
    axes2[1].scatter(I1_bar_pred- 3, (J_pred-1)**2, edgecolors='red', facecolors='none', alpha=0.3)
    axes2[1].scatter(I1_bar_train- 3, (J_train-1)**2, edgecolors='black', facecolors='none', alpha=0.5)
    # axes2[1].scatter(z_i1- 3, (z_j-1)**2, color='blue', marker='x', s=50)
    axes2[1].set_xlabel(r'$\bar{I}_1 - 3$')
    axes2[1].set_ylabel(r'$(J - 1)^2$')
    axes2[1].set_title(r'$\bar{I}_1 - 3$ vs $(J - 1)^2$')

    # I2_bar vs J
    axes2[2].scatter(I2_bar_pred- 3, (J_pred-1)**2, edgecolors='red', facecolors='none', alpha=0.3)
    axes2[2].scatter(I2_bar_train- 3, (J_train-1)**2, edgecolors='black', facecolors='none', alpha=0.5)
    # axes2[2].scatter(z_i2- 3, (z_j-1)**2, color='blue', marker='x', s=50)
    axes2[2].set_xlabel(r'$\bar{I}_2 - 3$')
    axes2[2].set_ylabel(r'$(J - 1)^2$')
    axes2[2].set_title(r'$\bar{I}_2 - 3$ vs $(J - 1)^2$')
    fig2.suptitle("Training and testing invariant space", fontsize=20)
    fig2.tight_layout()
    fig2.savefig(os.path.join(save_path, "invariant_space_coverage.pdf"), dpi=600)



import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import os

def plot_disp_field(node_coords, cells, u_true, u_pred_mean, u_pred_std, save_path):
    # --- Color Definitions ---
    triangulation = tri.Triangulation(node_coords[:, 0], node_coords[:, 1], cells)
    
    # Calculate magnitudes and coordinates
    # Note: Using your transformed parameters for displacement application
    def get_mag(u): return np.linalg.norm(u, axis=1)
    
    coords_true = node_coords + u_true
    coords_pred = node_coords + u_pred_mean
    
    mag_true = get_mag(u_true)
    mag_pred = get_mag(u_pred_mean)
    error = np.linalg.norm(u_true - u_pred_mean, axis=1)
    mag_std = get_mag(u_pred_std) if u_pred_std.ndim > 1 else u_pred_std

    fig, axes = plt.subplots(2, 2, figsize=(8, 12))
    plt.suptitle('Deformed Field: Accuracy & Uncertainty', fontsize=20)

    # 1,1: Deformed domain wrt u_true
    tri_true = tri.Triangulation(coords_true[:, 0], coords_true[:, 1], cells)
    im1 = axes[0, 0].tripcolor(tri_true, mag_true, cmap='Blues')
    axes[0, 0].set_title('True Material')
    fig.colorbar(im1, ax=axes[0, 0])

    # 1,2: Deformed domain wrt u_pred
    tri_pred = tri.Triangulation(coords_pred[:, 0], coords_pred[:, 1], cells)
    im2 = axes[0, 1].tripcolor(tri_pred, mag_pred, cmap='Blues')
    axes[0, 1].set_title('Predicted Material')
    fig.colorbar(im2, ax=axes[0, 1])

    # 2,1: Nodal error on pred deformed domain
    im3 = axes[1, 0].tripcolor(tri_pred, error, cmap='inferno')
    axes[1, 0].set_title('$(||u_{true} - u_{pred}||)$')
    fig.colorbar(im3, ax=axes[1, 0])

    # 2,2: u_std on pred deformed domain
    im4 = axes[1, 1].tripcolor(tri_pred, mag_std, cmap='magma')
    axes[1, 1].set_title('Uncertainty $(\sigma_u)$')
    fig.colorbar(im4, ax=axes[1, 1])

    # Standardize labels
    for ax in axes.flat:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        ax.label_outer()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "displacement_analysis.pdf"), dpi=300, bbox_inches='tight')
# Usage:
# plot_force_fields(node_coords, cells, R_nodes)
def parse_args():
    parser = argparse.ArgumentParser(description="Isihara Model Dataset and Training Configuration")

    # Dataset & Model Config
    parser.add_argument('--model_path', type=str, default="20260411T115941_isihara_0.0_0.01_8_0.975_5_40.0_1_0")
    # parser.add_argument('--validation_load_step_indices', type=int, nargs='+', default=[2, 4, 6, 8])
    parser.add_argument('--distilled_dir', type=str, required=True)
    parser.add_argument('--material_model', type=str, required=True)
    parser.add_argument('--n_sample', type=int, default=512)
    parser.add_argument('--subfolder', type=str, default="fem_validation")
    parser.add_argument('--geometry', type=str, default="block")
    parser.add_argument('--target_load', type=float, default=None)

    return parser.parse_args()
if __name__ == "__main__" :
    args = parse_args()
    target_total_samples = args.n_sample
    analysis_dir = Path("validation/coverage_test") 
    case_name = args.model_path if args.model_path else args.distilled_dir
    folder_name = os.path.basename(os.path.normpath(case_name))
    parent_folder_name = os.path.basename(os.path.dirname(os.path.normpath(case_name)))
    
    if folder_name.isdigit() or len(folder_name.split("_")) < 4:
        folder_name = parent_folder_name

    dataset_params = folder_name.split("_")
    
    if len(dataset_params) >= 6 and dataset_params[4].replace('.', '', 1).isdigit() and dataset_params[5].replace('.', '', 1).isdigit():
        material_model_name = dataset_params[1]
        disp_noise = float(dataset_params[2])
        load_noise = float(dataset_params[3])
        target_load = float(dataset_params[4]) if args.target_load is None else args.target_load
        asym_factor = float(dataset_params[5])
    else:
        material_model_name = dataset_params[1] if len(dataset_params) > 1 else "nh2"
        disp_noise = float(dataset_params[2]) if len(dataset_params) > 2 and dataset_params[2].replace('.', '', 1).isdigit() else 0.0001
        load_noise = float(dataset_params[3]) if len(dataset_params) > 3 and dataset_params[3].replace('.', '', 1).isdigit() else 0.01
        target_load = 1.5 if args.target_load is None else args.target_load
        asym_factor = 0.95

    print(f"Loaded model configuration: model={material_model_name}, disp_noise={disp_noise}, load_noise={load_noise}, target_load={target_load}, asym={asym_factor}")

    save_path = Path(args.distilled_dir) / args.subfolder
    save_path.mkdir(parents=True, exist_ok=True)
    # get I_obs_all.npy

    from core.material_models import get_material_from_dir
    true_material_model = get_material_from_dir(args.distilled_dir)
    true_piola_stress_func = lambda f : true_material_model.P(fto3x3(f))[:2, :2]

    # Define constitutive relationship.
    class HyperElasticity(Problem):
        def __init__(self, material_model_piola_stress, **kwargs) :
            super().__init__(**kwargs)
            self.material_model_piola_stress = material_model_piola_stress # should be function outputing piola stress [2x2 matrix]
        def custom_init(self):
            self.fe = self.fes[0]
        def get_surface_maps(self):
            def surface_map_top(u, x, load):
                return jnp.array([0., -load[0]])
            def surface_map_right(u,x,load) :
                if geometry_flag == "holes":
                    return jnp.array([load[0], 0.0])
                else:
                    return jnp.array([-load[0], 0.0])
            return [surface_map_right, surface_map_top]
        def set_params(self, params):
            surface_params = params
            self.internal_vars_surfaces = [[surface_params]]
        def get_tensor_map(self):

            def first_PK_stress(u_grad):
                I = jnp.eye(self.dim)
                F = u_grad + I
                P = self.material_model_piola_stress(F)
                return P

            return first_PK_stress

    geometry_flag = args.geometry

    u_exp = None
    prep_dataset_path = None
    for search_dir in ["dataset/preprocessed/syn_f", "dataset/precomputed_vfm"]:
        if os.path.exists(search_dir):
            for fname in os.listdir(search_dir):
                if fname.startswith(f"{material_model_name}_{disp_noise}_{load_noise}_{target_load}"):
                    if geometry_flag == "block" and ("_holes" not in fname):
                        if fname.endswith(".npz"):
                            prep_dataset_path = os.path.join(search_dir, fname)
                            break
                    elif geometry_flag != "block" and (f"_{geometry_flag}" in fname):
                        prep_dataset_path = os.path.join(search_dir, fname)
                        break
        if prep_dataset_path is not None:
            break

    if prep_dataset_path is not None:
        try:
            prep_data = np.load(prep_dataset_path, allow_pickle=True)
            if "u_exp" in prep_data:
                u_exp = prep_data["u_exp"]
            elif "u_true" in prep_data:
                u_exp = prep_data["u_true"]
            else:
                u_exp = prep_data["u"]
            
            # Load mesh directly from dataset
            node_coords = prep_data["mesh_pos"][:, :2] if "mesh_pos" in prep_data else prep_data["node_coords"][:, :2]
            cells = prep_data["cells"]
            node_type = prep_data["node_type"]
        except Exception as e:
            print(f"Failed to load u_exp from {prep_dataset_path}: {e}")
            sys.exit(1)
    else:
        print(f"Warning: No dataset found for geometry={geometry_flag}. Defaulting to mesh/{geometry_flag}_mesh.npz")
        mesh_data = jnp.load(f"mesh/{geometry_flag}_mesh.npz")
        node_coords = mesh_data["node_coords"][:, :2]
        cells = mesh_data["cells"]
        node_type = np.zeros(node_coords.shape[0], dtype=int)

    ele_type = 'TRI3'
    cell_type = get_meshio_cell_type(ele_type)
    data_dir = os.path.join('data')

    mesh = Mesh(node_coords, cells)

    # Define boundary locations.
    def left(point):
        return jnp.isclose(point[0], 0., atol=1e-6)
    def bottom(point):
        return jnp.isclose(point[1], 0., atol=1e-6)
    def right(point):
        return jnp.isclose(point[0], 1., atol=1e-6)
    def top(point):
        return jnp.isclose(point[1], 1.0, atol=1e-6)

    zero_dbc = lambda point : 0
    top_dbc = lambda point : 0.1

    if geometry_flag == "holes":
        dirichlet_bc_info = [
            [bottom, bottom] , 
            [0, 1],
            [zero_dbc, zero_dbc]]
    else:
        dirichlet_bc_info = [
            [bottom, left] , 
            [1, 0],
            [zero_dbc, zero_dbc]]
        
        # If node_type was not loaded from dataset, rebuild it for block
        if "prep_dataset_path" not in locals() or prep_dataset_path is None:
            node_type[jax.vmap(left)(node_coords)] = 1
            node_type[jax.vmap(bottom)(node_coords)] = 2
            node_type[jax.vmap(right)(node_coords)] = 3
            node_type[jax.vmap(top)(node_coords)] = 4

    consolidated_file = os.path.join(save_path, "fem_distilled_samples.npz")
    existing_u_pred = None
    existing_selected_samples = None
    num_existing = 0

    if os.path.exists(consolidated_file):
        try:
            existing_data = np.load(consolidated_file, allow_pickle=True)
            existing_u_pred = existing_data["u_pred"]
            existing_selected_samples = existing_data["selected_samples"]
            num_existing = existing_u_pred.shape[0]
            print(f"Found existing {num_existing} FEM sample realizations in {consolidated_file}.")
        except Exception as e:
            print(f"Could not load existing consolidated file: {e}. Starting fresh.")

    n_needed = target_total_samples - num_existing
    if n_needed <= 0:
        print(f"Target sample count ({target_total_samples}) already reached (current count: {num_existing}). Exiting.")
        n_sample = 0
    else:
        n_sample = n_needed
        print(f"Generating {n_sample} new samples to reach target {target_total_samples} total samples...")

    if n_sample > 0:
        flow_samples_path = os.path.join(args.distilled_dir, "flow_samples.npy")
        dev_samples_path = os.path.join(args.distilled_dir, "dev_flow_samples.npy")
        vol_samples_path = os.path.join(args.distilled_dir, "vol_flow_samples.npy")
        
        np.random.seed(42 + num_existing)
        if os.path.exists(dev_samples_path) and os.path.exists(vol_samples_path):
            dev_samples = np.load(dev_samples_path)
            vol_samples = np.load(vol_samples_path)
            num_total = min(len(dev_samples), len(vol_samples))
            dev_indices = np.random.choice(len(dev_samples), num_total, replace=False)
            vol_indices = np.random.choice(len(vol_samples), num_total, replace=False)
            selected_samples = np.concatenate([dev_samples[dev_indices], vol_samples[vol_indices]], axis=1)
        else:
            flow_samples = np.load(flow_samples_path)
            num_total = len(flow_samples)
            sample_indices = np.random.choice(len(flow_samples), num_total, replace=False)
            selected_samples = flow_samples[sample_indices]
        
        true_mat_model = true_material_model
        psi_true_func = lambda f: true_mat_model.psi(f)
        piola_true_func = lambda f: true_mat_model.P(f)

        problem_true = HyperElasticity(mesh=mesh,
                                vec=2,
                                dim=2,
                                ele_type=ele_type,
                                dirichlet_bc_info=dirichlet_bc_info,
                                location_fns=[right, top],
                                material_model_piola_stress=true_piola_stress_func)

        petsc_options = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_monitor": None,
            "snes_atol": 1e-10,
            "snes_rtol": 1e-10,
            "snes_stol": 1e-10,
            "snes_max_it": 50,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
        key = jax.random.PRNGKey(42 + num_existing)
        num_steps = 10
        noise_std = load_noise * target_load
        target_load_noisy = target_load + noise_std * jax.random.normal(key)

        noisy_load_top_base = jnp.linspace(0.0, target_load_noisy, num_steps).reshape(-1, 1)
        if geometry_flag == "holes":
            noisy_load_right_base = jnp.zeros_like(noisy_load_top_base)
        else:
            noisy_load_right_base = noisy_load_top_base * asym_factor
        loads_noisy = jnp.concat([noisy_load_right_base, noisy_load_top_base], axis=1)

        def solve_fem(problem, petsc_options, loads):
            u_list = []
            u = jnp.zeros_like(problem.mesh[0].points)
            for i, load in enumerate(loads):
                shape_right = (len(problem.boundary_inds_list[0]), problem.fes[0].num_face_quads, 1)
                shape_top = (len(problem.boundary_inds_list[1]), problem.fes[0].num_face_quads, 1)

                problem.internal_vars_surfaces = [
                    [
                        jnp.full(fill_value=load[0], shape=shape_right),
                    ],
                    [
                        jnp.full(fill_value=load[1], shape=shape_top)
                    ]
                ]
                u_ = solver(problem, solver_options={'petsc_solver': petsc_options,
                                                        'initial_guess': u})
                u = u_[0]
                u_list.append(u)
            u_array = jnp.stack(u_list, axis=0)  
            return u_array

        gt_dir = os.path.join(save_path, "gt")
        gt_file = os.path.join(gt_dir, "u_gt.npz")
        if os.path.exists(gt_file):
            u_true = np.load(gt_file)["u"]
        else:
            os.makedirs(gt_dir, exist_ok=True)
            u_true = solve_fem(problem_true, petsc_options, loads_noisy)
            np.savez_compressed(gt_file, u=u_true, cells=cells, node_coords=node_coords, node_type=node_type)

        u_pred_samples = []
        actual_selected_samples = []
        main_key = jr.PRNGKey(128 + num_existing)
        
        sample_idx = 0
        success_count = 0
        
        while success_count < n_sample and sample_idx < len(selected_samples):
            params = selected_samples[sample_idx]
            sample_idx += 1
            
            from core.material_models import HyperelasticModel
            angles = getattr(true_material_model, "angles", None)
            a0 = getattr(true_material_model, "a0", None)
            a1 = getattr(true_material_model, "a1", None)
            
            # Generalized loading for 10 dev + 3 vol + N aniso
            dev = list(params[:10])
            vol = list(params[10:13]) if len(params) >= 13 else list(params[len(dev):len(dev)+3])
            aniso = list(params[13:]) if len(params) > 13 else None
            
            # Pad to ensure expected lengths for HyperelasticModel
            dev = dev + [0.0] * (10 - len(dev))
            vol = vol + [0.0] * (3 - len(vol))
            
            mat = HyperelasticModel(dev_params=dev, vol_params=vol, aniso_params=aniso, angles=angles, a0=a0, a1=a1)

                
            problem_pred = HyperElasticity(
                mesh=mesh,
                vec=2,
                dim=2,
                ele_type=ele_type,
                dirichlet_bc_info=dirichlet_bc_info,
                location_fns=[right, top],
                material_model_piola_stress=lambda f: mat.P(fto3x3(f))[:2, :2]
            )
            
            try:
                print(f"Sample {num_existing + success_count + 1}/{target_total_samples}: Attempting realization {sample_idx}/{len(selected_samples)}...")
                u_pred = solve_fem(problem_pred, petsc_options, loads_noisy)
                success = True 
            except Exception as e:
                print(f"Simulation failed on realization {sample_idx}: {e}")
                success = False

            if success:
                u_pred_samples.append(u_pred)
                actual_selected_samples.append(params)
                success_count += 1
                print(f"Sample {num_existing + success_count} completed successfully.")

        if len(u_pred_samples) > 0:
            new_u_arr = np.array(u_pred_samples) # Shape: (n_sample, n_steps, n_nodes, 2)
            actual_selected_samples = np.array(actual_selected_samples)
            if existing_u_pred is not None:
                combined_u = np.concatenate([existing_u_pred, new_u_arr], axis=0)
                combined_params = np.concatenate([existing_selected_samples, actual_selected_samples], axis=0)
            else:
                combined_u = new_u_arr
                combined_params = actual_selected_samples

            save_dict = {
                "u_pred": combined_u,
                "selected_samples": combined_params,
                "node_coords": node_coords,
                "cells": cells,
                "node_type": node_type,
                "loads": loads_noisy
            }
            if 'u_true' in locals() and u_true is not None:
                save_dict["u_true"] = u_true
            if u_exp is not None:
                save_dict["u_exp"] = u_exp

            np.savez_compressed(consolidated_file, **save_dict)
            print(f"🎉 Consolidated FEM dataset updated: total {combined_u.shape[0]} samples saved to {consolidated_file}")