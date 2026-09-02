
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
import argparse
from sklearn.metrics import r2_score

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
    parser.add_argument('--n_sample', type=int, default=128)


    return parser.parse_args()
if __name__ == "__main__" :
    args = parse_args()
    # validation_load_step_indices = args.validation_load_step_indices
    n_sample = args.n_sample
    # load result 
    analysis_dir = Path("validation/coverage_test") 
    extraction_result_dir = Path("extraction/extracted_models") 
    # case_name = f"20260410T172507_isihara_0.0_0.01_8_0.975_5_40_0_0"
    case_name = args.model_path
    dataset_params = case_name.split("_")
    material_model_name = dataset_params[1]
    asym_factor = float(dataset_params[5])
    target_load = float(dataset_params[4])
    load_noise = float(dataset_params[3])
    save_path = analysis_dir / Path(args.distilled_dir).name
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
        
    # Specify mesh-related information (first-order hexahedron element).
    mesh_data = jnp.load("mesh/block_mesh.npz")
    node_coords = mesh_data["node_coords"][:, :2]
    cells = mesh_data["cells"]
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
    dirichlet_bc_info = [
        [bottom, left] , 
        [1, 0],
        [zero_dbc, zero_dbc]]
    
    node_type = np.zeros(node_coords.shape[0], dtype=int)
    check = jnp.sum(jax.vmap(left)(node_coords))
    node_type[jax.vmap(left)(node_coords)] = 1
    node_type[jax.vmap(bottom)(node_coords)] = 2
    node_type[jax.vmap(right)(node_coords)] = 3
    node_type[jax.vmap(top)(node_coords)] = 4

    flow_samples_path = os.path.join(args.distilled_dir, "flow_samples.npy")
    dev_samples_path = os.path.join(args.distilled_dir, "dev_flow_samples.npy")
    vol_samples_path = os.path.join(args.distilled_dir, "vol_flow_samples.npy")
    
    np.random.seed(42)
    if os.path.exists(dev_samples_path) and os.path.exists(vol_samples_path):
        dev_samples = np.load(dev_samples_path)
        vol_samples = np.load(vol_samples_path)
        dev_indices = np.random.choice(len(dev_samples), n_sample, replace=False)
        vol_indices = np.random.choice(len(vol_samples), n_sample, replace=False)
        selected_samples = np.concatenate([dev_samples[dev_indices], vol_samples[vol_indices]], axis=1)
    else:
        flow_samples = np.load(flow_samples_path)
        sample_indices = np.random.choice(len(flow_samples), n_sample, replace=False)
        selected_samples = flow_samples[sample_indices]
    
    
    true_mat_model = get_material(material_model_name)
    psi_true_func = lambda f: true_mat_model.psi(f)
    piola_true_func = lambda f: true_mat_model.P(f)


    # # Create an instance of the problem.
    problem_true = HyperElasticity(mesh = mesh,
                            vec=2,
                            dim=2,
                            ele_type=ele_type,
                            dirichlet_bc_info=dirichlet_bc_info,
                            location_fns = [right,top],
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
    key = jax.random.PRNGKey(42)

    # 2. Calculate Noise Scale (1% of average load)
    # We take the mean of all non-zero load magnitudes to define the noise floor

    # avg_load = jnp.mean(jnp.abs(loads_base)) 
    # noise_std = load_noise * avg_load
    # loads_noisy = loads_base + noise_std * jax.random.normal(key, shape=(num_steps, 2))

    # We want a final shape of (128, 10, 2)
    # We broadcast the (10, 2) base across 128 samples and add noise
    # noise = jax.random.normal(key, shape=(num_load_samples, num_steps, 2)) * noise_std
    # loads_noisy = jnp.expand_dims(loads_base, axis=0) + noise
    
    # loads_right = loads_top * asym_factor
    # loads = jnp.stack([loads_right, loads_top], axis=1)
    # Solve the defined problem.

    # u_true = jnp.zeros_like(node_coords)

    def solve_fem(problem, petsc_options, loads) :
        u_list = []
        u = jnp.zeros_like(problem.mesh[0].points)
        for i, load in enumerate(loads):
            print("load step ", i, "= ", load)
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
            u_= solver(problem, solver_options={'petsc_solver': petsc_options,
                                                    'initial_guess': u})
            u = u_[0]
            u_list.append(u)
        u_array = jnp.stack(u_list, axis=0)  
        return u_array
    
    num_steps = 10
    # loads_top = jnp.linspace(0.0, 10, 10)
    noise_std = load_noise * target_load
    target_load_noisy = target_load + noise_std * jax.random.normal(key)

    # noisy_target_loads = 
    noisy_load_top_base = jnp.linspace(0.0, target_load_noisy, num_steps).reshape(-1,1)
    noisy_load_right_base = noisy_load_top_base * asym_factor
    # Baseline loads shape: (10, 2)
    loads_noisy = jnp.concat([noisy_load_right_base, noisy_load_top_base], axis=1)
    if material_model_name == "neohookean" or material_model_name == "gentthomas":
        pass # loads_noisy = loads_noisy[:-2]


    u_true = solve_fem(problem_true, petsc_options, loads_noisy)
    u_pred_samples = []



    # pred_piola_stress_funcs = []
    main_key = jr.PRNGKey(128)
    # piola_keys = jr.split(main_key, n_piola_sample)
    # for key in piola_keys :
    #     pred_piola_stress_func = lambda f: model.piola(fto3x3(f), key)[:2, :2]
    #     pred_piola_stress_funcs.append(pred_piola_stress_func)

    for i in range(n_sample):
        success = False
        tries = 0
        max_tries = 5
        
        while not success and tries < max_tries:
            main_key, subkey = jr.split(main_key)
            
            sample_dir = os.path.join(save_path, "piola_traction_samples")
            save_file = os.path.join(sample_dir, f"u_pred_ps{i}.npz")
            if os.path.exists(save_file):
                try:
                    loaded_data = np.load(save_file)
                    if loaded_data['u_pred'].shape[0] == 10:
                        u_pred = loaded_data['u_pred']
                        success = True
                        break
                except:
                    pass

            main_key, key_traction, key_piola = jr.split(main_key, 3)
            
            # 1. Generate Noisy Loads
            noise_std = load_noise * target_load
            target_load_noisy = target_load + noise_std * jax.random.normal(key_traction)

            noisy_load_top_base = jnp.linspace(0.0, target_load_noisy, num_steps).reshape(-1, 1)
            noisy_load_right_base = noisy_load_top_base * asym_factor
            loads_noisy = jnp.concat([noisy_load_right_base, noisy_load_top_base], axis=1)
            if material_model_name == "neohookean" or material_model_name == "gentthomas":
                pass # loads_noisy = loads_noisy[:-2]
            # 2. Setup Problem
            params = selected_samples[i]
            if args.material_model == "isihara":
                c10, c01, c20, d1 = params[:4]
                mat = get_material(args.material_model, c10=c10, c01=c01, c20=c20, d1=d1)
            elif args.material_model in ["gmr_aniso", "aniso_gmr"]:
                dev = params[:10]
                vol = params[10:13]
                aniso = params[13:]
                angles = getattr(true_material_model, "angles", None)
                a0 = getattr(true_material_model, "a0", None)
                a1 = getattr(true_material_model, "a1", None)
                mat = get_material("gmr_aniso", dev_params=dev, vol_params=vol, aniso_params=aniso, angles=angles, a0=a0, a1=a1)
            elif args.material_model in ["gmr", "gmr_log", "gmr_nolog"]:
                if len(params) == 13: # 9 dev + 1 log + 3 vol
                    dev = params[:10]
                    vol = params[10:13]
                elif len(params) >= 14:
                    dev = params[:11]
                    vol = params[11:14]
                else: # 9 dev + 3 vol (no log term)
                    dev = params[:9]
                    vol = params[9:12]
                mat = get_material(args.material_model, dev_params=dev, vol_params=vol)
            else:
                dev = params[:min(10, len(params))]
                vol = params[min(10, len(params)):min(13, len(params))]
                mat = get_material(args.material_model, dev_params=dev, vol_params=vol)
                
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
                print(f"Sample {i}: Attempt {tries + 1}/{max_tries}...")
                # 3. Attempt Solve
                u_pred = solve_fem(problem_pred, petsc_options, loads_noisy)
                success = True # Mark success to break the while loop
                
            except Exception as e:
                tries += 1
                print(f"Solver failed on sample {i} (Attempt {tries}): {e}")
                if tries >= max_tries:
                    print(f"CRITICAL: Failed to solve sample {i} after {max_tries} attempts.")
                    # You can choose to 'break' to skip this sample or 'raise' to stop the script
                    raise e

        # 4. Save results only if successful
        if success:
            u_pred_samples.append(u_pred)
            
            out_dir = os.path.join(save_path, "piola_traction_samples")
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
                
            np.savez_compressed(
                os.path.join(out_dir, f"u_pred_ps{i}.npz"), 
                u_pred=u_pred, 
                cells=cells, 
                node_coords=node_coords, 
                node_type=node_type
            )
            print(f"Sample {i} successfully stored.")
    u_pred_sample_array = jnp.array(u_pred_samples)[:, -1, :, :]

    u_pred_mean = u_pred_sample_array.mean(axis=0)
    u_pred_std = u_pred_sample_array.std(axis=0)