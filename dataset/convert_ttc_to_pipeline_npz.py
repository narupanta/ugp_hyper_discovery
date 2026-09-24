"""
dataset/convert_ttc_to_pipeline_npz.py: Convert experimental TTc full-field DIC dataset
into the standardized pipeline .npz archive format for unsupervised GP discovery and FEM simulation.
"""

import os
import argparse
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

# Enforce 64-bit precision standard
jax.config.update("jax_enable_x64", True)

from core.utils import deformation_gradient_element
from core.fem_engine import make_plane_stress_piola
from core.material_models import get_material


def parse_args():
    parser = argparse.ArgumentParser(description="Convert TTc experimental dataset to pipeline .npz format")
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="dataset/experiment/TTc/TTc/dataset_600_steps",
        help="Path to directory containing experimental step folders"
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[6000],
        help="List of load step numbers to include (e.g., 6000 or 1000 3000 6000)"
    )
    parser.add_argument(
        "--plane_stress_mode",
        type=str,
        default="compressible",
        choices=["compressible", "incompressible"],
        help="Plane stress formulation: 'compressible' (solves P_33=0) or 'incompressible' (lambda3 = 1 / det(F_2D))"
    )
    parser.add_argument(
        "--compressible_model",
        type=str,
        default="gentthomas",
        choices=["gentthomas", "neohookean"],
        help="Material model for solving P_33=0 in compressible plane stress"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="dataset/preprocessed/experiment",
        help="Output directory for converted .npz file"
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default="",
        help="Output filename (defaults to ttc_step_<steps>.npz)"
    )
    parser.add_argument(
        "--mesh_out",
        type=str,
        default="mesh/ttc_mesh.npz",
        help="Output path for reference mesh"
    )
    return parser.parse_args()


def convert_dataset(
    exp_dir: str,
    steps: list,
    out_path: str,
    mesh_out: str,
    plane_stress_mode: str = "compressible",
    compressible_model: str = "gentthomas"
):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if mesh_out:
        os.makedirs(os.path.dirname(os.path.abspath(mesh_out)), exist_ok=True)

    print(f"[CONVERT] Converting TTc experimental data from {exp_dir}")
    print(f"[CONVERT] Target load steps: {steps}")
    print(f"[CONVERT] Plane stress mode: {plane_stress_mode} (model: {compressible_model if plane_stress_mode == 'compressible' else 'N/A'})")

    if plane_stress_mode == "compressible":
        if compressible_model == "gentthomas":
            mat_model = get_material(
                "gentthomas",
                dev_params=[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                vol_params=[1.5, 0.0, 0.0]
            )
        elif compressible_model == "neohookean":
            mat_model = get_material("neohookean", c1=0.5, c2=1.5)
        else:
            raise ValueError(f"Unknown compressible model: {compressible_model}")
        _, solve_lambda3_fn = make_plane_stress_piola(mat_model, max_iter=15)
        solve_lambda3_vmap = jax.vmap(solve_lambda3_fn)
    else:
        solve_lambda3_vmap = None

    # Read reference mesh from first specified step
    ref_step_dir = os.path.join(exp_dir, str(steps[0]))
    if not os.path.exists(ref_step_dir):
        raise FileNotFoundError(f"Step directory not found: {ref_step_dir}")

    df_nodes_ref = pd.read_csv(os.path.join(ref_step_dir, "output_nodes.csv"))
    df_elems_ref = pd.read_csv(os.path.join(ref_step_dir, "output_elements.csv"))
    df_integ_ref = pd.read_csv(os.path.join(ref_step_dir, "output_integrator.csv"))

    mesh_pos = df_nodes_ref[["x", "y"]].values.astype(np.float64)  # (N_nodes, 2)
    cells = df_elems_ref[["node1", "node2", "node3"]].values.astype(np.int64)  # (N_elems, 3)
    bcx = df_nodes_ref["bcx"].values.astype(np.int32)
    bcy = df_nodes_ref["bcy"].values.astype(np.int32)

    n_nodes = mesh_pos.shape[0]
    n_elems = cells.shape[0]
    print(f"[CONVERT] Reference mesh: {n_nodes} nodes, {n_elems} triangular elements.")

    # Construct 5-channel node_type:
    # Channel 0: is_internal (bcx == 0)
    # Channel 1: is_fix_x (bcx == 1, left clamp)
    # Channel 2: is_fix_y (bcx == 1, left clamp)
    # Channel 3: is_loaded_x (bcx == 2, right clamp)
    # Channel 4: is_loaded_y (bcx == 2, right clamp)
    node_type = np.zeros((n_nodes, 5), dtype=np.float32)
    node_type[:, 0] = (bcx == 0).astype(np.float32)
    node_type[:, 1] = (bcx == 1).astype(np.float32)
    node_type[:, 2] = (bcx == 1).astype(np.float32)
    node_type[:, 3] = (bcx == 2).astype(np.float32)
    node_type[:, 4] = (bcx == 2).astype(np.float32)

    print(f"[CONVERT] Boundary nodes: bcx=1 (left clamp): {np.sum(bcx == 1)}, bcx=2 (right clamp): {np.sum(bcx == 2)}, internal/free: {np.sum(bcx == 0)}")

    # Compute reference geometry element areas (dA) and shape function gradients (dNdX)
    coords_elems = jnp.array(mesh_pos[cells])  # (n_elems, 3, 2)
    
    # Compute using element kinematics
    disp_zeros = jnp.zeros_like(coords_elems)
    _, dNdX_jax = deformation_gradient_element(coords_elems, disp_zeros)
    dNdX = np.array(dNdX_jax, dtype=np.float64)  # (n_elems, 3, 2)

    # Area of linear triangles: 0.5 * |det J|
    x1, y1 = coords_elems[:, 0, 0], coords_elems[:, 0, 1]
    x2, y2 = coords_elems[:, 1, 0], coords_elems[:, 1, 1]
    x3, y3 = coords_elems[:, 2, 0], coords_elems[:, 2, 1]
    detJ = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    dA = np.array(0.5 * jnp.abs(detJ), dtype=np.float64)

    # Validate against output_integrator.csv if available
    if "qpWeight" in df_integ_ref:
        qp_weights = df_integ_ref["qpWeight"].values
        diff_area = np.max(np.abs(dA - qp_weights))
        print(f"[CONVERT] Maximum absolute difference between computed dA and qpWeight: {diff_area:.2e}")

    # Collect time-series data across load steps
    u_list = []
    load_list = []
    reaction_forces_list = []
    F_list = []
    F_3d_list = []
    lam3_list = []

    for step_num in steps:
        step_dir = os.path.join(exp_dir, str(step_num))
        if not os.path.exists(step_dir):
            raise FileNotFoundError(f"Load step directory not found: {step_dir}")

        df_nodes = pd.read_csv(os.path.join(step_dir, "output_nodes.csv"))
        df_elems = pd.read_csv(os.path.join(step_dir, "output_elements.csv"))
        df_reac = pd.read_csv(os.path.join(step_dir, "output_reactions.csv"))

        # Displacements
        u_step = df_nodes[["ux", "uy"]].values.astype(np.float64)  # (N_nodes, 2)
        u_list.append(u_step)

        # Reaction forces: forces[0]=left rx, forces[1]=right rx
        forces = df_reac["forces"].values
        rx_left = float(forces[0])
        rx_right = float(forces[1])
        load_list.append([rx_right, 0.0])
        reaction_forces_list.append([rx_right, 0.0])

        # Deformation gradient computation
        disp_elems = jnp.array(u_step[cells])  # (n_elems, 3, 2)
        f_2x2, _ = deformation_gradient_element(coords_elems, disp_elems)
        f_2x2_np = np.array(f_2x2, dtype=np.float64)  # (n_elems, 2, 2)
        F_list.append(f_2x2_np)

        det_f2d = np.linalg.det(f_2x2_np)
        if plane_stress_mode == "compressible":
            lam3_jax = solve_lambda3_vmap(f_2x2)
            lam3 = np.array(lam3_jax, dtype=np.float64)
        else:
            det_f2d_safe = np.clip(det_f2d, 1e-6, 1e6)
            lam3 = 1.0 / det_f2d_safe
        lam3_list.append(lam3)

        # Construct 3x3 deformation gradient
        f_3x3 = np.zeros((n_elems, 3, 3), dtype=np.float64)
        f_3x3[:, :2, :2] = f_2x2_np
        f_3x3[:, 2, 2] = lam3
        F_3d_list.append(f_3x3)

        J_step = det_f2d * lam3
        print(f"  Step {step_num}: Reaction Rx_right = {rx_right:.3f} N, mean stretch ratio J_2D = {np.mean(det_f2d):.4f}, lambda3 = [{np.min(lam3):.4f}, {np.mean(lam3):.4f}, {np.max(lam3):.4f}], J = [{np.min(J_step):.4f}, {np.mean(J_step):.4f}, {np.max(J_step):.4f}]")

    u_arr = np.stack(u_list, axis=0)                # (N_steps, N_nodes, 2)
    load_arr = np.array(load_list, dtype=np.float64) # (N_steps, 2)
    reaction_arr = np.array(reaction_forces_list, dtype=np.float64) # (N_steps, 2)
    F_arr = np.stack(F_list, axis=0)                # (N_steps, N_elems, 2, 2)
    F_3d_arr = np.stack(F_3d_list, axis=0)          # (N_steps, N_elems, 3, 3)
    lam3_arr = np.stack(lam3_list, axis=0)          # (N_steps, N_elems)

    n_steps = len(steps)
    f_neu_arr = np.zeros((n_steps, n_nodes, 2), dtype=np.float64)
    load_noise_std = np.maximum(np.abs(load_arr) * 0.01, 1e-4)
    load_noise_std_steps = np.copy(load_noise_std)

    # Save comprehensive pipeline .npz archive
    np.savez_compressed(
        out_path,
        mesh_pos=mesh_pos,
        cells=cells,
        node_type=node_type,
        dNdX=dNdX,
        dA=dA,
        u=u_arr,
        u_obs=u_arr,
        u_exp=u_arr,
        F=F_arr,
        F_3d=F_3d_arr,
        lam3=lam3_arr,
        load=load_arr,
        reaction_forces=reaction_arr,
        f_neu=f_neu_arr,
        load_noise_std=load_noise_std,
        load_noise_std_steps=load_noise_std_steps,
        control_mode=np.array("displacement"),
        stress_mode=np.array("plane_stress"),
        plane_stress_mode=np.array(plane_stress_mode),
        has_ground_truth=np.array(False),
        steps=np.array(steps),
        bcx=bcx,
        bcy=bcy
    )
    print(f"[CONVERT] Successfully saved pipeline dataset: {out_path}")

    # Also save reference mesh for forward FEM simulation if requested
    if mesh_out:
        np.savez_compressed(
            mesh_out,
            node_coords=mesh_pos,
            cells=cells,
            node_type=node_type,
            bcx=bcx,
            bcy=bcy
        )
        print(f"[CONVERT] Successfully saved reference mesh: {mesh_out}")


def main():
    args = parse_args()
    if not args.out_name:
        if len(args.steps) == 1:
            out_name = f"ttc_step_{args.steps[0]}.npz"
        else:
            out_name = f"ttc_{len(args.steps)}_steps.npz"
    else:
        out_name = args.out_name
    out_path = os.path.join(args.out_dir, out_name)
    convert_dataset(
        args.exp_dir,
        args.steps,
        out_path,
        args.mesh_out,
        plane_stress_mode=args.plane_stress_mode,
        compressible_model=args.compressible_model
    )


if __name__ == "__main__":
    main()
