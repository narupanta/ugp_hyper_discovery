
# Import some useful modules.
import jax
import jax.numpy as jnp
import sys
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
from core.fem_engine import (
    get_geometry,
    create_default_bc_config,
    HyperElasticityProblem,
    solve_adaptive_fem,
    compute_all_invariants
)
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
    parser.add_argument('--worker_id', type=int, default=0, help="Worker ID for parallel chunking (0-indexed)")
    parser.add_argument('--total_workers', type=int, default=1, help="Total number of parallel workers")
    parser.add_argument('--sample_offset', type=int, default=0, help="Starting index in candidate samples pool")
    parser.add_argument('--output_suffix', type=str, default="", help="Optional suffix for worker output file")
    parser.add_argument('--output_dir', type=str, default=None, help="Direct output directory for FEM validation")
    parser.add_argument('--dataset_path', type=str, default="", help="Explicit path to precomputed dataset npz file")
    parser.add_argument('--num_steps', type=int, default=None, help="Number of load steps (defaults to dataset load shape or 10)")
    parser.add_argument('--control_mode', type=str, default=None, choices=['force', 'displacement'], help="Boundary control mode ('force' or 'displacement')")
    parser.add_argument('--stress_mode', type=str, default=None, choices=['plane_strain', 'plane_stress'], help="Stress state assumption ('plane_strain' or 'plane_stress')")
    parser.add_argument('--prescribe_right', type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=None, help="Prescribe right boundary")
    parser.add_argument('--clamp_top_x', type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=None, help="Clamp top boundary in X direction")
    parser.add_argument('--asym_factor', type=float, default=None, help="Asymmetry factor between top and right boundaries")

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

    if args.output_dir:
        save_path = Path(args.output_dir)
    else:
        save_path = Path(args.distilled_dir) / args.subfolder
    save_path.mkdir(parents=True, exist_ok=True)
    # get I_obs_all.npy

    from core.material_models import get_material_from_dir
    true_material_model = get_material_from_dir(args.distilled_dir)

    geometry_flag = args.geometry

    u_exp = None
    prep_dataset_path = None
    if args.dataset_path and os.path.exists(args.dataset_path):
        prep_dataset_path = os.path.abspath(args.dataset_path)
        print(f"[VAL] Using explicit dataset path: {prep_dataset_path}")
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        
        search_dirs = [
            os.path.join(project_root, "dataset/preprocessed/syn_f"),
            os.path.join(project_root, "dataset/precomputed_vfm"),
            "dataset/preprocessed/syn_f",
            "dataset/precomputed_vfm"
        ]
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for fname in sorted(os.listdir(search_dir)):
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
            print(f"[VAL] Found matching dataset: {prep_dataset_path}")

    prep_data = None
    u_true = None
    if prep_dataset_path is not None:
        try:
            prep_data = np.load(prep_dataset_path, allow_pickle=True)
            if "u_true" in prep_data:
                u_true = prep_data["u_true"]
            elif "u" in prep_data:
                u_true = prep_data["u"]

            if "u_exp" in prep_data:
                u_exp = prep_data["u_exp"]
            elif "u_obs" in prep_data:
                u_exp = prep_data["u_obs"]
            else:
                u_exp = u_true
            
            # Load mesh directly from dataset
            node_coords = prep_data["mesh_pos"][:, :2] if "mesh_pos" in prep_data else prep_data["node_coords"][:, :2]
            cells = prep_data["cells"]
            node_type = prep_data["node_type"]
        except Exception as e:
            print(f"Failed to load dataset from {prep_dataset_path}: {e}")
            sys.exit(1)
    else:
        print(f"Warning: No dataset found for {material_model_name}_{disp_noise}_{load_noise}_{target_load} with geometry={geometry_flag}.")
        print(f"Searched in: {search_dirs}")
        print(f"Defaulting to mesh/{geometry_flag}_mesh.npz")
        mesh_path = os.path.join(project_root, f"mesh/{geometry_flag}_mesh.npz")
        if not os.path.exists(mesh_path):
            print(f"CRITICAL ERROR: Fallback mesh {mesh_path} not found! Cannot proceed with FEM.")
            sys.exit(1)
        mesh_data = jnp.load(mesh_path)
        node_coords = mesh_data["node_coords"][:, :2]
        cells = mesh_data["cells"]
        node_type = np.zeros(node_coords.shape[0], dtype=int)

    # Determine control_mode and stress_mode
    control_mode = args.control_mode
    if control_mode is None and prep_data is not None and "control_mode" in prep_data:
        control_mode = str(prep_data["control_mode"])
    if control_mode is None:
        control_mode = "force"
    control_mode = control_mode.lower()

    stress_mode = args.stress_mode
    if stress_mode is None and prep_data is not None and "stress_mode" in prep_data:
        stress_mode = str(prep_data["stress_mode"])
    if stress_mode is None:
        stress_mode = "plane_strain"
    stress_mode = stress_mode.lower()

    prescribe_right = args.prescribe_right
    if prescribe_right is None:
        prescribe_right = (geometry_flag != "holes")

    if args.asym_factor is not None:
        asym_factor = args.asym_factor

    clamp_top_x = args.clamp_top_x
    if clamp_top_x is None:
        if "clamp_top_x" in rec:
            clamp_top_x = bool(rec["clamp_top_x"])
        else:
            clamp_top_x = (geometry_flag == "holes" and control_mode == "displacement")
    if geometry_flag != "holes":
        clamp_top_x = False

    print(f"[VAL] Configured validation modes: control_mode={control_mode}, stress_mode={stress_mode}, prescribe_right={prescribe_right}, clamp_top_x={clamp_top_x}")

    # Set up ground truth stress function
    if stress_mode == "plane_stress":
        from core.fem_engine import make_plane_stress_piola
        true_piola_stress_func, solve_lambda3_true = make_plane_stress_piola(true_material_model)
    else:
        true_piola_stress_func = lambda f: true_material_model.P(fto3x3(f))[:2, :2]
        solve_lambda3_true = None

    # Distilled hyperelastic energy and stress functions
    def eval_psi_3d(F_3d, p, a0=None, a1=None):
        C = C_func(F_3d)
        I1 = I1_func(C)
        I2 = I2_func(C)
        I3 = I3_func(C)
        I3_safe = jnp.clip(I3, 1.0e-8, 1.0e8)

        i1_dev = I3_safe ** (-1 / 3) * I1
        i2_dev = I3_safe ** (-2 / 3) * I2
        J = jnp.sqrt(I3_safe)
        i1_m3 = i1_dev - 3.0
        i2_m3 = i2_dev - 3.0
        J_m1 = J - 1.0

        dev_p = p[:10]
        vol_p = p[10:13]

        w_dev = (
            dev_p[0] * i1_m3 + dev_p[1] * i2_m3 + dev_p[2] * i1_m3**2 +
            dev_p[3] * i1_m3 * i2_m3 + dev_p[4] * i2_m3**2 + dev_p[5] * i1_m3**3 +
            dev_p[6] * (i1_m3**2) * i2_m3 + dev_p[7] * i1_m3 * (i2_m3**2) +
            dev_p[8] * i2_m3**3 + dev_p[9] * jnp.log(jnp.maximum(i2_dev / 3.0, 1e-8))
        )
        w_vol = vol_p[0] * J_m1**2 + vol_p[1] * J_m1**4 + vol_p[2] * J_m1**6

        w_aniso = 0.0
        if a0 is not None and len(p) > 13:
            C_bar = (I3_safe ** (-1 / 3))[..., None, None] * C
            I4_bar = jnp.einsum('i,...ij,j->...', a0, C_bar, a0)
            I4_m1 = I4_bar - 1.0
            aniso_p = p[13:19]
            w_aniso = aniso_p[0] * I4_m1**2 + aniso_p[1] * I4_m1**3 + aniso_p[2] * I4_m1**4
            if a1 is not None and len(p) > 16:
                I6_bar = jnp.einsum('i,...ij,j->...', a1, C_bar, a1)
                I6_m1 = I6_bar - 1.0
                w_aniso += aniso_p[3] * I6_m1**2 + aniso_p[4] * I6_m1**3 + aniso_p[5] * I6_m1**4

        return w_dev + w_vol + w_aniso

    def p_3d_distilled(F_3d, p, a0=None, a1=None):
        return jax.grad(eval_psi_3d, argnums=0)(F_3d, p, a0, a1)

    def solve_lambda3_distilled(F_2d, p, a0=None, a1=None, max_iter=8):
        det_2d = jnp.linalg.det(F_2d)
        lam3_0 = 1.0 / jnp.clip(det_2d, 1e-4, 1e4)

        def step_fn(i, lam):
            def p33_val(l):
                F_3d = jnp.array([
                    [F_2d[0, 0], F_2d[0, 1], 0.0],
                    [F_2d[1, 0], F_2d[1, 1], 0.0],
                    [0.0,        0.0,        l]
                ])
                return p_3d_distilled(F_3d, p, a0, a1)[2, 2]

            p33, dp33 = jax.value_and_grad(p33_val)(lam)
            lam_next = lam - p33 / jnp.where(jnp.abs(dp33) < 1e-12, 1.0, dp33)
            return jnp.clip(lam_next, 1e-3, 100.0)

        return jax.lax.fori_loop(0, max_iter, step_fn, lam3_0)

    def piola_stress_distilled(F_2d, p, a0=None, a1=None):
        if stress_mode == "plane_stress":
            lam3 = solve_lambda3_distilled(F_2d, p, a0, a1)
            F_3d = jnp.array([
                [F_2d[0, 0], F_2d[0, 1], 0.0],
                [F_2d[1, 0], F_2d[1, 1], 0.0],
                [0.0,        0.0,        lam3]
            ])
            return p_3d_distilled(F_3d, p, a0, a1)[:2, :2]
        else: # plane_strain
            F_3d = jnp.array([
                [F_2d[0, 0], F_2d[0, 1], 0.0],
                [F_2d[1, 0], F_2d[1, 1], 0.0],
                [0.0,        0.0,        1.0]
            ])
            return p_3d_distilled(F_3d, p, a0, a1)[:2, :2]

    ele_type = 'TRI3'
    cell_type = get_meshio_cell_type(ele_type)
    data_dir = os.path.join('data')

    mesh = Mesh(node_coords, cells)

    # Boundary Condition Setup
    geom = get_geometry(geometry_flag)
    bc_config = create_default_bc_config(
        geometry_name=geometry_flag,
        mode=control_mode,
        pred_dict=geom.get_boundary_predicates(),
        prescribe_right=prescribe_right,
        clamp_top_x=clamp_top_x
    )
    dirichlet_bc_info = bc_config.get_dirichlet_info()
    surface_maps = bc_config.get_surface_maps()
    location_fns = [nbc.location_fn for nbc in bc_config.neumann_bcs]
    node_type = bc_config.create_node_type_array(node_coords)

    # Determine load / displacement schedule and step count
    if control_mode == "force":
        if prep_data is not None and "load" in prep_data:
            loads_solve = jnp.array(prep_data["load"])
            num_steps = loads_solve.shape[0]
            schedule_solve = loads_solve
            print(f"[VAL] Using force loading schedule from dataset with {num_steps} load steps.")
        else:
            key = jax.random.PRNGKey(42)
            num_steps = args.num_steps if getattr(args, "num_steps", None) is not None else 10
            noise_std = load_noise * target_load
            target_load_noisy = target_load + noise_std * jax.random.normal(key)

            noisy_load_top_base = jnp.linspace(0.0, target_load_noisy, num_steps).reshape(-1, 1)
            if geometry_flag == "holes" or not prescribe_right:
                noisy_load_right_base = jnp.zeros_like(noisy_load_top_base)
            else:
                noisy_load_right_base = noisy_load_top_base * asym_factor
            loads_solve = jnp.concat([noisy_load_right_base, noisy_load_top_base], axis=1)
            schedule_solve = loads_solve
            print(f"[VAL] Generated synthetic linspace loading schedule with {num_steps} load steps.")
        observed_loads = loads_solve
    else: # displacement control
        if u_true is not None:
            num_steps = u_true.shape[0]
            top_mask = jax.vmap(geom.get_boundary_predicates()["top"])(node_coords)
            disps_top_true = u_true[:, top_mask, 1].mean(axis=1, keepdims=True)
            if prescribe_right:
                right_mask = jax.vmap(geom.get_boundary_predicates()["right"])(node_coords)
                disps_right_true = u_true[:, right_mask, 0].mean(axis=1, keepdims=True)
                schedule_solve = jnp.concatenate([disps_right_true, disps_top_true], axis=1)
            else:
                schedule_solve = disps_top_true
            print(f"[VAL] Extracted exact prescribed displacement schedule from u_true ({num_steps} steps).")
        else:
            num_steps = args.num_steps if getattr(args, "num_steps", None) is not None else 10
            disps_top_true = jnp.linspace(0.0, target_load, num_steps).reshape(-1, 1)
            if prescribe_right:
                disps_right_true = disps_top_true * asym_factor
                schedule_solve = jnp.concatenate([disps_right_true, disps_top_true], axis=1)
            else:
                schedule_solve = disps_top_true
            print(f"[VAL] Generated synthetic linspace displacement schedule ({num_steps} steps, target={target_load}).")

        if prep_data is not None and "load" in prep_data:
            observed_loads = jnp.array(prep_data["load"])
        elif prep_data is not None and "reaction_forces" in prep_data:
            observed_loads = jnp.array(prep_data["reaction_forces"])
        else:
            observed_loads = np.zeros((num_steps, 2))
        loads_solve = observed_loads

    if args.output_suffix:
        file_name = f"fem_distilled_samples_{args.output_suffix}.npz"
    elif args.total_workers > 1:
        file_name = f"fem_distilled_samples_worker{args.worker_id}.npz"
    else:
        file_name = "fem_distilled_samples.npz"
    consolidated_file = os.path.join(save_path, file_name)
    existing_u_pred = None
    existing_selected_samples = None
    num_existing = 0

    if os.path.exists(consolidated_file):
        try:
            existing_data = np.load(consolidated_file, allow_pickle=True)
            existing_u_pred = existing_data["u_pred"]
            existing_selected_samples = existing_data["selected_samples"]
            cached_ctrl = str(existing_data["control_mode"]) if "control_mode" in existing_data else "force"
            cached_stress = str(existing_data["stress_mode"]) if "stress_mode" in existing_data else "plane_strain"
            if existing_u_pred.shape[1] != num_steps:
                print(f"[VAL] Warning: Cached u_pred has {existing_u_pred.shape[1]} steps, but expected {num_steps} steps. Discarding stale cache.")
                existing_u_pred = None
                existing_selected_samples = None
                num_existing = 0
            elif cached_ctrl != control_mode or cached_stress != stress_mode:
                print(f"[VAL] Warning: Cached FEM mode mismatch (cached: {cached_ctrl}/{cached_stress} vs current: {control_mode}/{stress_mode}). Discarding stale cache.")
                existing_u_pred = None
                existing_selected_samples = None
                num_existing = 0
            else:
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
        
        # Fixed master seed so block and holes draw the exact same realizations without replacement
        np.random.seed(42)
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
        
        # Filter out already existing samples to guarantee no duplicate parameter realizations
        if existing_selected_samples is not None and len(existing_selected_samples) > 0:
            unique_candidates = []
            for s in selected_samples:
                # Check if this parameter candidate was already evaluated
                is_duplicate = any(np.allclose(s, ex, atol=1e-7) for ex in existing_selected_samples)
                if not is_duplicate:
                    unique_candidates.append(s)
            selected_samples = np.array(unique_candidates)
            print(f"Filtered out {len(existing_selected_samples)} already evaluated realizations. {len(selected_samples)} candidates remaining.")

        # Parallel chunking support across workers
        if args.total_workers > 1:
            worker_chunk_size = int(np.ceil(target_total_samples / args.total_workers))
            start_idx = args.worker_id * worker_chunk_size
            end_idx = min(start_idx + worker_chunk_size, len(selected_samples))
            selected_samples = selected_samples[start_idx:end_idx]
            n_sample = len(selected_samples)
            print(f"Worker {args.worker_id}/{args.total_workers}: Assigned candidate slice [{start_idx}:{end_idx}] ({n_sample} samples).")
        elif args.sample_offset > 0:
            selected_samples = selected_samples[args.sample_offset:args.sample_offset + n_sample]
            print(f"Worker with offset {args.sample_offset}: Evaluating {len(selected_samples)} samples.")

        angles = getattr(true_material_model, "angles", None)
        a0 = getattr(true_material_model, "a0", None)
        a1 = getattr(true_material_model, "a1", None)

        # Helper to convert parameter sequence to fixed 19-vector (10 dev + 3 vol + 6 aniso)
        def to_19_param_vec(p_seq):
            p_arr = np.zeros(19, dtype=np.float64)
            p_len = min(len(p_seq), 19)
            p_arr[:p_len] = p_seq[:p_len]
            return jnp.array(p_arr)

        problem_true = HyperElasticityProblem(
            mesh=mesh,
            vec=2,
            dim=2,
            ele_type=ele_type,
            dirichlet_bc_info=dirichlet_bc_info,
            location_fns=location_fns,
            surface_maps=surface_maps,
            piola_func=lambda F: true_piola_stress_func(F[:2, :2] if F.shape == (3, 3) else F)
        )

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
        gt_dir = os.path.join(save_path, "gt")
        gt_file = os.path.join(gt_dir, "u_gt.npz")
        u_true_cached = None
        if os.path.exists(gt_file):
            try:
                cached_data = np.load(gt_file)
                if cached_data["u"].shape[0] == num_steps:
                    u_true_cached = cached_data["u"]
                    print(f"[VAL] Loaded existing ground truth from {gt_file}")
                else:
                    print(f"[VAL] Warning: Cached ground truth has {cached_data['u'].shape[0]} steps != {num_steps}. Re-evaluating.")
            except Exception as e:
                print(f"[VAL] Error reading gt_file: {e}")

        if u_true_cached is not None:
            u_true = u_true_cached
        elif u_true is not None and u_true.shape[0] == num_steps:
            os.makedirs(gt_dir, exist_ok=True)
            np.savez_compressed(gt_file, u=u_true, cells=cells, node_coords=node_coords, node_type=node_type)
            print(f"[VAL] Reused and cached ground truth displacements from dataset: shape={u_true.shape}")
        else:
            os.makedirs(gt_dir, exist_ok=True)
            print(f"[VAL] Solving adaptive FEM for ground truth with {num_steps} steps (schedule shape: {schedule_solve.shape})...")
            u_true = solve_adaptive_fem(problem_true, bc_config, schedule_solve, petsc_options)
            np.savez_compressed(gt_file, u=u_true, cells=cells, node_coords=node_coords, node_type=node_type)

        u_pred_samples = []
        actual_selected_samples = []
        main_key = jr.PRNGKey(128 + num_existing)
        
        import time
        t_fem_start = time.time()
        
        sample_idx = 0
        success_count = 0

        # Instantiate problem_pred ONCE outside the sample loop
        problem_pred = HyperElasticityProblem(
            mesh=mesh,
            vec=2,
            dim=2,
            ele_type=ele_type,
            dirichlet_bc_info=dirichlet_bc_info,
            location_fns=location_fns,
            surface_maps=surface_maps,
            num_internal_params=19,
            piola_func=lambda F, p: piola_stress_distilled(F[:2, :2] if F.shape == (3, 3) else F, p, a0, a1)
        )
        
        while success_count < n_sample and sample_idx < len(selected_samples):
            params = selected_samples[sample_idx]
            sample_idx += 1

            # Update parameters dynamically without re-instantiating Problem or triggering JIT recompilation
            p_vec = to_19_param_vec(params)
            problem_pred.set_params(p_vec)
            
            try:
                print(f"Sample {num_existing + success_count + 1}/{target_total_samples}: Attempting realization {sample_idx}/{len(selected_samples)}...")
                u_pred = solve_adaptive_fem(problem_pred, bc_config, schedule_solve, petsc_options)
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

            t_fem_duration = float(time.time() - t_fem_start)
            save_dict = {
                "u_pred": combined_u,
                "selected_samples": combined_params,
                "node_coords": node_coords,
                "cells": cells,
                "node_type": node_type,
                "loads": observed_loads,
                "schedule_solve": np.array(schedule_solve),
                "control_mode": control_mode,
                "stress_mode": stress_mode,
                "fem_time_sec": t_fem_duration
            }
            if 'u_true' in locals() and u_true is not None:
                save_dict["u_true"] = u_true
            if u_exp is not None:
                save_dict["u_exp"] = u_exp
            if prep_data is not None and "lam3" in prep_data:
                save_dict["lam3"] = prep_data["lam3"]
            if prep_data is not None and "lam3_true" in prep_data:
                save_dict["lam3_true"] = prep_data["lam3_true"]

            np.savez_compressed(consolidated_file, **save_dict)
            print(f"🎉 Consolidated FEM dataset updated: total {combined_u.shape[0]} samples (time: {t_fem_duration:.2f}s) saved to {consolidated_file}")