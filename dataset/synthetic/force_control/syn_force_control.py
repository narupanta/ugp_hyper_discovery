import os
import sys
import argparse
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from jax_fem.generate_mesh import Mesh
from core.utils import fto3x3
from core.material_models import get_material

from core.fem_engine import (
    get_geometry,
    create_default_bc_config,
    HyperElasticityProblem,
    solve_adaptive_fem,
    export_fem_dataset
)

import matplotlib.pyplot as plt
import matplotlib.tri as tri

def plot_dataset_viz(data, dataset_name, save_path):
    mesh_pos = data["mesh_pos"]
    cells = data["cells"]
    ux = np.array(data["u"][-1, :, 0])
    uy = np.array(data["u"][-1, :, 1])
    u = np.column_stack((ux, uy))
    world_pos = mesh_pos[:, :2] + u

    x = mesh_pos[:, 0]
    y = mesh_pos[:, 1]
    triangulation = tri.Triangulation(x, y, cells)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Finite Element Visualization (Undeformed vs. Deformed)', fontsize=16)

    ax1 = axes[0]
    tpc1 = ax1.tripcolor(triangulation, ux, cmap='viridis', edgecolors='k', linewidth=0.5)
    fig.colorbar(tpc1, ax=ax1, label='$u_x$ Displacement')
    ax1.set_title('Color Plot: $u_x$ (Horizontal Displacement)')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_aspect('equal')

    ax2 = axes[1]
    tpc2 = ax2.tripcolor(triangulation, uy, cmap='magma', edgecolors='k', linewidth=0.01)
    fig.colorbar(tpc2, ax=ax2, label='$u_y$ Displacement')
    ax2.set_title('Color Plot: $u_y$ (Vertical Displacement)')
    ax2.set_xlabel('X Position')
    ax2.set_aspect('equal')

    ax3 = axes[2]
    ax3.triplot(triangulation, 'r-', alpha=0.5, linewidth=0.5, label='Undeformed Mesh')
    x_def = world_pos[:, 0]
    y_def = world_pos[:, 1]
    tri_def = tri.Triangulation(x_def, y_def, cells)
    ax3.triplot(tri_def, 'b-', linewidth=0.5, label='Deformed Mesh')
    ax3.set_title('Deformed Domain')
    ax3.set_xlabel('X Position')
    ax3.legend()
    ax3.set_aspect('equal')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{dataset_name}.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Universal Modular FEM Dataset Generator")
    parser.add_argument('--model', type=str, default="isihara")
    parser.add_argument('--disp_noise', type=float, default=0.0)
    parser.add_argument('--load_noise', type=float, default=0.03)
    parser.add_argument('--target_top', type=float, default=10.0)
    parser.add_argument('--asym', type=float, default=0.9)
    parser.add_argument('--n_steps', type=int, default=21)
    parser.add_argument('--seed', type=int, default=42, help="Random seed for data generation")
    parser.add_argument('--mesh_dir', type=str, default="mesh")
    parser.add_argument('--raw_data_dir', type=str, default="dataset/synthetic/force_control")
    parser.add_argument('--precomputed_dir', type=str, default="dataset/preprocessed/syn_f")
    parser.add_argument('--geometry', type=str, default='block')
    parser.add_argument('--mesh_size', type=float, default=0.08)
    parser.add_argument('--control_mode', type=str, default="force", choices=["force", "displacement"])
    parser.add_argument('--angles', type=float, nargs='+', default=None)
    parser.add_argument('--dev_params', type=float, nargs='+', default=None)
    parser.add_argument('--vol_params', type=float, nargs='+', default=None)
    parser.add_argument('--aniso_params', type=float, nargs='+', default=None)
    args = parser.parse_args()

    material_model_name = args.model
    disp_noise = args.disp_noise
    load_noise = args.load_noise
    target_load = args.target_top
    asym_factor = args.asym
    num_steps = args.n_steps
    mesh_dir = args.mesh_dir
    raw_data_dir = args.raw_data_dir
    precomputed_dir = args.precomputed_dir
    geometry_name = args.geometry.lower()
    control_mode = args.control_mode.lower()

    os.makedirs(mesh_dir, exist_ok=True)
    mesh_msh_path = os.path.join(mesh_dir, f"{geometry_name}_mesh.msh")
    mesh_npz_path = os.path.join(mesh_dir, f"{geometry_name}_mesh.npz")

    # 1. Geometry & Mesh Generation
    geom = get_geometry(geometry_name, mesh_size=args.mesh_size)
    if not os.path.exists(mesh_npz_path):
        geom.generate_mesh(mesh_msh_path, mesh_npz_path)

    mesh_data = np.load(mesh_npz_path)
    node_coords = mesh_data["node_coords"][:, :2]
    cells = mesh_data["cells"]
    ele_type = 'TRI3'
    mesh = Mesh(node_coords, cells)

    # 2. Boundary Condition Configuration
    pred_dict = geom.get_boundary_predicates()
    bc_config = create_default_bc_config(geometry_name=geometry_name, mode=control_mode, pred_dict=pred_dict)
    dirichlet_bc_info = bc_config.get_dirichlet_info()
    surface_maps = bc_config.get_surface_maps()
    node_type = bc_config.create_node_type_array(node_coords)

    # 3. Material Constitutive Model
    mat_kwargs = {}
    if args.angles is not None: mat_kwargs["angles"] = args.angles
    if args.dev_params is not None: mat_kwargs["dev_params"] = args.dev_params
    if args.vol_params is not None: mat_kwargs["vol_params"] = args.vol_params
    if args.aniso_params is not None: mat_kwargs["aniso_params"] = args.aniso_params

    # If parameters not explicitly given on CLI, attempt to load defaults from recipe
    recipe_path = os.path.join("configs", "recipes", f"{material_model_name}.yaml")
    if os.path.exists(recipe_path):
        import yaml
        with open(recipe_path, "r") as f:
            rec = yaml.safe_load(f)
        mat_p = rec.get("material_params", {})
        if "dev_params" not in mat_kwargs and "dev_params" in mat_p:
            mat_kwargs["dev_params"] = mat_p["dev_params"]
        if "vol_params" not in mat_kwargs and "vol_params" in mat_p:
            mat_kwargs["vol_params"] = mat_p["vol_params"]
        if "aniso_params" not in mat_kwargs and "aniso_params" in mat_p:
            mat_kwargs["aniso_params"] = mat_p["aniso_params"]
        if "angles" not in mat_kwargs and "angles" in mat_p:
            mat_kwargs["angles"] = mat_p["angles"]

    true_mat_model = get_material(material_model_name, **mat_kwargs)
    true_piola_stress_func = lambda f: true_mat_model.P(fto3x3(f))[:2, :2]

    # 4. Universal Problem Instance
    problem_true = HyperElasticityProblem(
        mesh=mesh,
        vec=2,
        dim=2,
        ele_type=ele_type,
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=[nbc.location_fn for nbc in bc_config.neumann_bcs],
        surface_maps=surface_maps,
        piola_func=true_piola_stress_func
    )

    # PETSc SNES solver options
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

    # 5. Schedule (Force or Displacement)
    key = jax.random.PRNGKey(args.seed)
    if control_mode == "force":
        noise_std = load_noise * target_load
        target_load_noisy = target_load + noise_std * jax.random.normal(key)

        noisy_load_top_base = jnp.linspace(0.0, target_load_noisy, num_steps).reshape(-1, 1)
        if geometry_name == "holes":
            noisy_load_right_base = jnp.zeros_like(noisy_load_top_base)
        else:
            noisy_load_right_base = noisy_load_top_base * asym_factor
        loads_noisy = jnp.concatenate([noisy_load_right_base, noisy_load_top_base], axis=1)

        loads_top_true = jnp.linspace(0.0, target_load, num_steps).reshape(-1, 1)
        if geometry_name == "holes":
            loads_right_true = jnp.zeros_like(loads_top_true)
        else:
            loads_right_true = loads_top_true * asym_factor
        loads_true = jnp.concatenate([loads_right_true, loads_top_true], axis=1)

        schedule_solve = loads_true
        load_noise_std = load_noise * loads_true
        load_noise_std_steps = load_noise_std * np.linspace(0, 1, num_steps).reshape(-1, 1)
    else: # displacement
        disps_top_true = jnp.linspace(0.0, target_load, num_steps).reshape(-1, 1)
        disps_right_true = disps_top_true * asym_factor
        schedule_solve = jnp.concatenate([disps_right_true, disps_top_true], axis=1)
        loads_noisy = jnp.zeros_like(schedule_solve)
        load_noise_std = np.zeros_like(schedule_solve)
        load_noise_std_steps = np.zeros_like(schedule_solve)

    # 6. Solve Adaptive FEM
    print(f"Solving forward FEM ({geometry_name}, {material_model_name}, {num_steps} steps, mode={control_mode})...")
    u_true = solve_adaptive_fem(problem_true, bc_config, schedule_solve, petsc_options)

    # Save per-step raw datasets
    save_raw_dataset_dir = os.path.join(raw_data_dir, f"{material_model_name}_{disp_noise}_{load_noise}_{target_load}_{asym_factor}_{args.seed}")
    os.makedirs(save_raw_dataset_dir, exist_ok=True)
    for step in range(u_true.shape[0]):
        step_data = {
            "mesh_pos": node_coords,
            "cells": cells,
            "u": u_true[step],
            "node_type": node_type,
            "load": loads_noisy[step],
            "load_noise_std": load_noise_std
        }
        np.savez_compressed(f"{save_raw_dataset_dir}/disp_{step:02d}.npz", **step_data)

    # 7. Comprehensive Dataset Exporter
    dataset_name = f"{material_model_name}_{disp_noise}_{load_noise}_{target_load}_{asym_factor}_{geometry_name}_{args.seed}"
    output_npz_path = os.path.join(precomputed_dir, f"{dataset_name}.npz")

    a0 = getattr(true_mat_model, 'a0', None)
    a1 = getattr(true_mat_model, 'a1', None)
    a2 = getattr(true_mat_model, 'a2', None)

    export_fem_dataset(
        output_npz_path=output_npz_path,
        mesh_pos=node_coords,
        cells=cells,
        node_type=node_type,
        u_true=np.array(u_true),
        loads_noisy=loads_noisy,
        disp_noise=disp_noise,
        load_noise_std=load_noise_std,
        load_noise_std_steps=load_noise_std_steps,
        seed=args.seed,
        a0=a0,
        a1=a1,
        a2=a2,
        mode=control_mode
    )

    # Create unseeded fallback copy for standard single-seed workflows if seed in [0, 42, 1]
    fallback_name = f"{material_model_name}_{disp_noise}_{load_noise}_{target_load}_{asym_factor}_{geometry_name}.npz"
    fallback_path = os.path.join(precomputed_dir, fallback_name)
    if not os.path.exists(fallback_path):
        import shutil
        shutil.copyfile(output_npz_path, fallback_path)

    # Diagnostic visual verification
    viz_data = dict(u=u_true, mesh_pos=node_coords, cells=cells, node_type=node_type)
    plot_dataset_viz(viz_data, dataset_name, "dataset_viz_jax")
    print(f"✅ Generated and exported dataset successfully to {output_npz_path}")


if __name__ == "__main__":
    main()
