import os
import argparse

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
    make_plane_stress_piola
)
from core.dataset_store import (
    CLEAN_DIR, clean_dataset_path, make_dataset_spec, save_clean_dataset, displacement_reactions
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
    parser.add_argument('--recipe', type=str, default=None, help="Path to recipe YAML configuration file")
    parser.add_argument('--model', type=str, default="isihara")
    parser.add_argument('--disp_noise', type=float, default=None)
    parser.add_argument('--load_noise', type=float, default=None)
    parser.add_argument('--target_top', type=float, default=None)
    parser.add_argument('--asym', type=float, default=None)
    parser.add_argument('--n_steps', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42, help="Seed of the noise realisation (the FEM solve is seed-independent)")
    parser.add_argument('--mesh_dir', type=str, default="mesh")
    parser.add_argument('--clean_dir', type=str, default=CLEAN_DIR, help="Directory of clean (noise-free) datasets")
    parser.add_argument('--spec_out', type=str, default=None,
                        help="Write the dataset spec (clean path + seed + noise levels) to this file")
    parser.add_argument('--spec_only', action='store_true',
                        help="Only resolve and write the spec of an existing clean dataset (no FEM solve)")
    parser.add_argument('--force_regen', action='store_true', help="Re-solve the FEM even if the clean dataset exists")
    parser.add_argument('--geometry', type=str, default='block')
    parser.add_argument('--mesh_size', type=float, default=0.08)
    parser.add_argument('--control_mode', type=str, default=None, choices=["force", "displacement"])
    parser.add_argument('--stress_mode', type=str, default=None, choices=["plane_strain", "plane_stress"])
    parser.add_argument('--prescribe_right', type=int, default=None, help="1 to prescribe right boundary, 0 to leave free. Default: 1 for block, 0 for holes.")
    parser.add_argument('--clamp_top_x', type=int, default=None, help="1 to clamp top boundary in x (ux=0, rigid clamp), 0 for free roller.")
    parser.add_argument('--angles', type=float, nargs='+', default=None)
    parser.add_argument('--dev_params', type=float, nargs='+', default=None)
    parser.add_argument('--vol_params', type=float, nargs='+', default=None)
    parser.add_argument('--aniso_params', type=float, nargs='+', default=None)
    args = parser.parse_args()

    material_model_name = args.model
    geometry_name = args.geometry.lower()

    # Load defaults from recipe if available
    recipe_path = args.recipe or os.path.join("configs", "recipes", f"{material_model_name}.yaml")
    rec = {}
    if os.path.exists(recipe_path):
        import yaml
        with open(recipe_path, "r") as f:
            rec = yaml.safe_load(f) or {}
        if "material_model_name" in rec and args.model == "isihara" and args.recipe is not None:
            material_model_name = rec["material_model_name"]

    control_mode = (args.control_mode or rec.get("control_mode", "force")).lower()
    stress_mode = (args.stress_mode or rec.get("stress_mode", "plane_strain")).lower()
    target_load = args.target_top if args.target_top is not None else float(rec.get("target_load_true_top", 10.0))
    asym_factor = args.asym if args.asym is not None else float(rec.get("asym_factor", 0.9))
    num_steps = args.n_steps if args.n_steps is not None else int(rec.get("n_loadsteps", 21))
    disp_noise = args.disp_noise if args.disp_noise is not None else float(rec.get("disp_noise", 0.0))
    load_noise = args.load_noise if args.load_noise is not None else float(rec.get("load_noise", 0.03))

    if args.clamp_top_x is not None:
        clamp_top_x_val = bool(args.clamp_top_x)
    elif "clamp_top_x" in rec:
        clamp_top_x_val = bool(rec["clamp_top_x"])
    else:
        clamp_top_x_val = (geometry_name == "holes" and control_mode == "displacement")

    if geometry_name != "holes":
        clamp_top_x_val = False
    if args.prescribe_right is not None:
        prescribe_right = bool(args.prescribe_right)
    else:
        prescribe_right = bool(rec.get("prescribe_right", (geometry_name != "holes")))

    mat_kwargs = {}
    if args.angles is not None: mat_kwargs["angles"] = args.angles
    if args.dev_params is not None: mat_kwargs["dev_params"] = args.dev_params
    if args.vol_params is not None: mat_kwargs["vol_params"] = args.vol_params
    if args.aniso_params is not None: mat_kwargs["aniso_params"] = args.aniso_params

    mat_p = rec.get("material_params", {})
    for k in ("dev_params", "vol_params", "aniso_params", "angles"):
        if k not in mat_kwargs and k in mat_p:
            mat_kwargs[k] = mat_p[k]

    # Everything that determines the noise-free FEM solution; its hash names the clean dataset.
    config = dict(
        material_model=material_model_name,
        material_kwargs={k: [float(x) for x in np.atleast_1d(v)] for k, v in mat_kwargs.items()},
        geometry=geometry_name, mesh_size=float(args.mesh_size),
        control_mode=control_mode, stress_mode=stress_mode,
        target_load=float(target_load), asym_factor=float(asym_factor), n_steps=int(num_steps),
        prescribe_right=bool(prescribe_right), clamp_top_x=bool(clamp_top_x_val),
    )
    clean_path = clean_dataset_path(config, root=args.clean_dir)
    spec = make_dataset_spec(clean_path, args.seed, disp_noise, load_noise)

    def write_spec():
        if args.spec_out:
            with open(args.spec_out, "w") as f:
                f.write(spec + "\n")
        print(f"DATASET_SPEC={spec}")

    if args.spec_only or (os.path.exists(clean_path) and not args.force_regen):
        if not os.path.exists(clean_path):
            raise FileNotFoundError(f"--spec_only: clean dataset {clean_path} does not exist; run generation first.")
        print(f"✅ Clean dataset exists, skipping FEM solve: {clean_path}")
        write_spec()
        return

    # 1. Geometry & Mesh Generation (cache keyed by geometry and mesh size)
    os.makedirs(args.mesh_dir, exist_ok=True)
    mesh_msh_path = os.path.join(args.mesh_dir, f"{geometry_name}_h{args.mesh_size}_mesh.msh")
    mesh_npz_path = os.path.join(args.mesh_dir, f"{geometry_name}_h{args.mesh_size}_mesh.npz")
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
    bc_config = create_default_bc_config(
        geometry_name=geometry_name,
        mode=control_mode,
        pred_dict=pred_dict,
        prescribe_right=prescribe_right,
        clamp_top_x=clamp_top_x_val
    )
    dirichlet_bc_info = bc_config.get_dirichlet_info()
    surface_maps = bc_config.get_surface_maps()
    node_type = bc_config.create_node_type_array(node_coords)

    # 3. Material Constitutive Model
    true_mat_model = get_material(material_model_name, **mat_kwargs)
    if stress_mode == "plane_stress":
        true_piola_stress_func, solve_lambda3 = make_plane_stress_piola(true_mat_model)
    else:
        true_piola_stress_func = lambda f: true_mat_model.P(fto3x3(f))[:2, :2]
        solve_lambda3 = None

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

    # 5. True loading schedule (noise is added at load time, see core.dataset_store.observe_dataset)
    ramp = jnp.linspace(0.0, target_load, num_steps).reshape(-1, 1)
    if control_mode == "force":
        right = jnp.zeros_like(ramp) if (geometry_name == "holes" or not prescribe_right) else ramp * asym_factor
        schedule_solve = jnp.concatenate([right, ramp], axis=1)
    else:
        schedule_solve = jnp.concatenate([ramp * asym_factor, ramp], axis=1) if prescribe_right else ramp

    # 6. Solve Adaptive FEM
    print(f"Solving forward FEM ({geometry_name}, {material_model_name}, {num_steps} steps, mode={control_mode})...")
    u_true = np.array(solve_adaptive_fem(problem_true, bc_config, schedule_solve, petsc_options))

    # 7. Clean dataset: true state only
    loads_true = np.array(schedule_solve)
    if loads_true.shape[1] == 1:
        loads_true = np.concatenate([np.zeros_like(loads_true), loads_true], axis=1)
    reactions_true = (displacement_reactions(u_true, node_coords, cells, node_type, true_piola_stress_func)
                      if control_mode == "displacement" else None)
    lam3_true = None
    if solve_lambda3 is not None:
        from core.utils import deformation_gradient_element
        F_true = [deformation_gradient_element(node_coords[cells], u_true[t][cells])[0] for t in range(num_steps)]
        lam3_true = np.array([np.array(jax.vmap(solve_lambda3)(F)) for F in F_true])

    save_clean_dataset(
        clean_path, config, node_coords, cells, node_type, u_true, loads_true,
        reaction_forces_true=reactions_true, lam3_true=lam3_true,
        a0=getattr(true_mat_model, 'a0', None), a1=getattr(true_mat_model, 'a1', None), a2=getattr(true_mat_model, 'a2', None),
    )

    # Diagnostic visual verification (once per clean dataset)
    viz_data = dict(u=u_true, mesh_pos=node_coords, cells=cells, node_type=node_type)
    plot_dataset_viz(viz_data, os.path.splitext(os.path.basename(clean_path))[0], "dataset_viz_jax")
    print(f"✅ Solved and stored clean dataset: {clean_path}")
    write_spec()


if __name__ == "__main__":
    main()
