#!/usr/bin/env python3
"""
plots/run_block_validation.py

Pre-FEM validation framework runner for the Block geometry.
Executes immediately following the distillation stage (before forward FEM simulation),
generating all equilibrium residual coverage plots, local probe distributions,
reaction force breakdowns, and step-by-step Empirical Coverage (EC) reports inside:
    <distilled_dir>/block_validation/

Outputs produced:
1. free_node_coverage_step19_exp_variance_comparison.pdf / .png
2. free_node_coverage_step19_true_variance_comparison.pdf / .png
3. reaction_force_block_all_steps_breakdown_obs.pdf / .png
4. local_node_residuals_step19_locations.pdf / .png
5. local_node_residuals_step19_exp_variance_comparison.pdf / .png
6. empirical_coverage_all_steps.md
7. free_node_coverage_all_steps_with_variance.json
8. Associated 4-panel global residual distribution and breakdown figures
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
from core.dataset_store import dataset_exists, load_dataset

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from plots.theme import apply_style
from plots.plot_reaction_force_distilled import (
    compute_distilled_reaction_forces
)
from plots.plot_reaction_force_block_full import plot_full_reaction_forces
from plots.plot_free_node_coverage import compute_residuals_and_coverage
from plots.plot_selected_node_residuals import plot_node_spatial_locations
from plots.plot_coverage_with_free_variance import (
    evaluate_coverage_with_and_without_variance,
    plot_four_panel_variant,
    plot_side_by_side_residuals_coverage_comparison,
    plot_local_nodes_variant,
    plot_local_nodes_variance_overlay
)
from plots.compute_all_steps_ec import compute_all_steps_ec


def resolve_data_bundle(distilled_dir: Path, extracted_dir: Path = None, dataset_path: str = None, save_path: Path = None):
    """
    Assembles or resolves the data bundle containing mesh, kinematics, and parameter samples.
    If fem_distilled_samples.npz is available in fem_validation, uses it.
    Otherwise, bundles dev/vol flow samples from distilled_dir with the dataset npz.
    """
    save_path.mkdir(parents=True, exist_ok=True)
    bundle_file = save_path / "block_validation_data.npz"

    # 1. Check if a pre-existing fem_distilled_samples.npz is already available
    cand_fem = distilled_dir.parent / "fem_validation" / "block" / "fem_distilled_samples.npz"
    if cand_fem.exists():
        print(f"ℹ️ Found existing FEM distilled samples bundle at: {cand_fem}")
        return str(cand_fem)

    if bundle_file.exists():
        print(f"ℹ️ Using existing block validation data bundle: {bundle_file}")
        return str(bundle_file)

    # 2. Otherwise assemble from distilled parameter samples and preprocessed dataset
    print("⏳ Assembling pre-FEM block validation data bundle from distillation and dataset...")
    dev_path = distilled_dir / "dev_flow_samples.npy"
    vol_path = distilled_dir / "vol_flow_samples.npy"
    if not dev_path.exists() or not vol_path.exists():
        raise FileNotFoundError(f"Distilled flow samples not found in {distilled_dir} (needed dev_flow_samples.npy & vol_flow_samples.npy)")

    dev_samples = np.load(dev_path)
    vol_samples = np.load(vol_path)
    n_samples = min(len(dev_samples), len(vol_samples))
    np.random.seed(42)
    dev_idx = np.random.choice(len(dev_samples), n_samples, replace=False)
    vol_idx = np.random.choice(len(vol_samples), n_samples, replace=False)
    selected_samples = np.concatenate([dev_samples[dev_idx], vol_samples[vol_idx]], axis=1)

    # Resolve dataset path
    ds_path = None
    if dataset_path and dataset_exists(dataset_path):
        ds_path = dataset_path
    else:
        # Check metadata.json in extracted dir
        if extracted_dir and (extracted_dir / "metadata.json").exists():
            try:
                with open(extracted_dir / "metadata.json", "r") as f:
                    meta = json.load(f)
                if "dataset_path" in meta and dataset_exists(meta["dataset_path"]):
                    ds_path = meta["dataset_path"]
            except Exception:
                pass

    if ds_path is None or not dataset_exists(str(ds_path)):
        # Search candidate datasets
        for cand_dir in [Path("dataset/preprocessed/syn_f"), Path("dataset/synthetic/force_control")]:
            if cand_dir.exists():
                for f in cand_dir.glob("*.npz"):
                    if "block" in f.name:
                        ds_path = f
                        break

    if ds_path is None or not dataset_exists(str(ds_path)):
        raise FileNotFoundError("Could not locate training dataset npz file for block geometry.")

    print(f"📂 Loading dataset from: {ds_path}")
    raw_data = load_dataset(str(ds_path))
    coords = raw_data["mesh_pos"][:, :2] if "mesh_pos" in raw_data else raw_data["node_coords"][:, :2]
    cells = raw_data["cells"]
    node_type = raw_data["node_type"]
    u_true = raw_data["u_true"] if "u_true" in raw_data else raw_data["u"]
    u_exp = raw_data["u_exp"] if "u_exp" in raw_data else (raw_data["u_obs"] if "u_obs" in raw_data else u_true)

    loads = raw_data["loads"] if "loads" in raw_data else (raw_data["load"] if "load" in raw_data else np.zeros((len(u_true), 2)))
    schedule_solve = raw_data["schedule_solve"] if "schedule_solve" in raw_data else loads
    control_mode = str(raw_data.get("control_mode", "displacement"))
    stress_mode = str(raw_data.get("stress_mode", "plane_stress"))

    bundle_dict = {
        "node_coords": coords,
        "cells": cells,
        "node_type": node_type,
        "selected_samples": selected_samples,
        "loads": loads,
        "schedule_solve": schedule_solve,
        "control_mode": control_mode,
        "stress_mode": stress_mode,
        "u_true": u_true,
        "u_exp": u_exp
    }
    np.savez_compressed(bundle_file, **bundle_dict)
    print(f"✅ Created pre-FEM block validation data bundle: {bundle_file}")
    return str(bundle_file)


def run_block_validation(
    distilled_dir: str,
    extracted_dir: str = None,
    dataset_path: str = None,
    save_path: str = None,
    step: int = 19,
    batch_size: int = 256,
    make_png: bool = True
):
    apply_style()
    dist_p = Path(distilled_dir).resolve()
    ext_p = Path(extracted_dir).resolve() if extracted_dir else (dist_p.parent / "extracted")
    out_p = Path(save_path).resolve() if save_path else (dist_p / "block_validation")
    out_p.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*75)
    print(f"🚀 RUNNING PRE-FEM BLOCK VALIDATION PIPELINE")
    print(f"   Distilled Directory: {dist_p}")
    print(f"   Extracted Directory: {ext_p}")
    print(f"   Target Save Path:    {out_p}")
    print(f"   Load Step Evaluated: {step}")
    print("="*75 + "\n")

    # 1. Resolve data bundle
    data_bundle_path = resolve_data_bundle(dist_p, ext_p, dataset_path, out_p)

    # 2. Resolve learned noise parameters (sigma_free_x, sigma_free_y)
    extraction_metrics_path = ext_p / "extraction_metrics.json"
    if not extraction_metrics_path.exists():
        # Fallback search
        for cand in [dist_p.parent / "extracted" / "extraction_metrics.json", dist_p / "extraction_metrics.json"]:
            if cand.exists():
                extraction_metrics_path = cand
                break

    if not extraction_metrics_path.exists():
        raise FileNotFoundError(f"extraction_metrics.json not found in {ext_p} or parent extracted directory.")

    with open(extraction_metrics_path, "r") as f:
        em = json.load(f)
    sigma_free_x = float(em.get("sigma_free_x", 0.01))
    sigma_free_y = float(em.get("sigma_free_y", 0.01))
    print(f"📈 Loaded learned residual noise: sigma_free_x = {sigma_free_x:.6f}, sigma_free_y = {sigma_free_y:.6f}")

    # Copy extraction_metrics.json into block_validation for standalone auditability
    try:
        import shutil
        shutil.copy2(extraction_metrics_path, out_p / "extraction_metrics.json")
    except Exception:
        pass

    # 3. Compute and cache reaction forces across all load steps
    cache_file = out_p / "reaction_forces_cache_block.npz"
    if not cache_file.exists():
        print("⏳ Computing reaction forces across all load steps for distilled samples...")
        rf_dict = compute_distilled_reaction_forces(data_bundle_path, batch_size=batch_size)
        
        # Also compute on clean true kinematics
        data_b = np.load(data_bundle_path, allow_pickle=True)
        if "u_true" in data_b:
            coords = jnp.array(data_b["node_coords"])
            cells = jnp.array(data_b["cells"])
            params = jnp.array(data_b["selected_samples"])
            node_type = np.array(data_b["node_type"])
            u_true = jnp.array(data_b["u_true"])
            stress_mode = str(data_b.get("stress_mode", "plane_stress")).lower()
            
            from core.utils import transformation_jacobian, deformation_gradient_element
            from plots.plot_reaction_force_distilled import piola_stress_2d
            coords_elems = coords[cells]
            is_fix_x = jnp.array(node_type[:, 1] == 1)
            is_fix_y = jnp.array(node_type[:, 2] == 1)
            J_mat = transformation_jacobian(coords_elems)
            dA = 0.5 * jnp.abs(jnp.linalg.det(J_mat))
            n_nodes = coords.shape[0]

            @jax.jit
            def compute_rx_ry_true(p):
                def step_fn(u_s):
                    disp_elems = u_s[cells]
                    F_c, dNdX = deformation_gradient_element(coords_elems, disp_elems)
                    P_c = jax.vmap(lambda f: piola_stress_2d(f, p, stress_mode=stress_mode))(F_c)
                    f_elem = jnp.einsum('cij,cnj->cni', P_c, dNdX) * dA[:, None, None]
                    f_int = jnp.zeros((n_nodes, 2), dtype=jnp.float64)
                    for a in range(3):
                        f_int = f_int.at[cells[:, a]].add(f_elem[:, a])
                    rx = -jnp.sum(jnp.where(is_fix_x[:, None], f_int, 0.0)[:, 0])
                    ry = -jnp.sum(jnp.where(is_fix_y[:, None], f_int, 0.0)[:, 1])
                    return rx, ry
                return jax.vmap(step_fn)(u_true)

            rx_true_list, ry_true_list = [], []
            n_samples = params.shape[0]
            for b in range(0, n_samples, batch_size):
                b_end = min(b + batch_size, n_samples)
                rx_b, ry_b = jax.vmap(compute_rx_ry_true)(params[b:b_end])
                rx_true_list.append(np.array(rx_b))
                ry_true_list.append(np.array(ry_b))
            rx_true_all = np.concatenate(rx_true_list, axis=0)
            ry_true_all = np.concatenate(ry_true_list, axis=0)
            rf_dict["rx_true_all"] = rx_true_all
            rf_dict["ry_true_all"] = ry_true_all

        # Determine gt reaction forces
        loads_arr = rf_dict["loads"]
        rf_dict["rx_gt"] = loads_arr[:, 0]
        rf_dict["ry_gt"] = loads_arr[:, 1]
        np.savez_compressed(cache_file, **rf_dict)
        print(f"✅ Saved reaction forces cache: {cache_file}")

    # 4. Generate Reaction Force Trajectory & Breakdown Plots
    print("📊 Generating Reaction Force verification plots and breakdown curves...")
    plot_full_reaction_forces(
        save_path=str(out_p),
        cache_file=str(cache_file),
        make_png=make_png
    )

    # 5. Compute base residuals and evaluate equilibrium coverage at step 19 (excluding boundary nodes)
    print(f"⏳ Evaluating free node equilibrium residuals at step {step} across 1024 samples...")
    res = compute_residuals_and_coverage(data_bundle_path, step=step, batch_size=batch_size)
    cov_data = evaluate_coverage_with_and_without_variance(
        res, sigma_free_x, sigma_free_y, n_mc_draws=100
    )

    # 6. Generate domain coverage comparison plots (Step 19)
    print("📊 Generating free node domain coverage comparison plots...")
    plot_side_by_side_residuals_coverage_comparison(res, cov_data, "exp", str(out_p))
    plot_side_by_side_residuals_coverage_comparison(res, cov_data, "true", str(out_p))

    # Also generate individual 4-panel figures
    plot_four_panel_variant(res, cov_data, "exp", "without_var", sigma_free_x, sigma_free_y, str(out_p))
    plot_four_panel_variant(res, cov_data, "exp", "with_var", sigma_free_x, sigma_free_y, str(out_p))
    plot_four_panel_variant(res, cov_data, "true", "without_var", sigma_free_x, sigma_free_y, str(out_p))
    plot_four_panel_variant(res, cov_data, "true", "with_var", sigma_free_x, sigma_free_y, str(out_p))

    # 7. Generate selected local node residual distribution plots and location map
    selected_nodes = [9, 10, 91, 114]
    print(f"📊 Generating local node distribution plots and location map for representative nodes {selected_nodes}...")
    plot_node_spatial_locations(res, selected_nodes, str(out_p))
    plot_local_nodes_variant(res, cov_data, selected_nodes, "exp", "without_var", sigma_free_x, sigma_free_y, str(out_p))
    plot_local_nodes_variant(res, cov_data, selected_nodes, "exp", "with_var", sigma_free_x, sigma_free_y, str(out_p))
    plot_local_nodes_variance_overlay(res, cov_data, selected_nodes, "exp", str(out_p))
    plot_local_nodes_variant(res, cov_data, selected_nodes, "true", "without_var", sigma_free_x, sigma_free_y, str(out_p))
    plot_local_nodes_variant(res, cov_data, selected_nodes, "true", "with_var", sigma_free_x, sigma_free_y, str(out_p))
    plot_local_nodes_variance_overlay(res, cov_data, selected_nodes, "true", str(out_p))

    # 8. Compute all-steps Empirical Coverage table & JSON
    print("📊 Computing 20-step Empirical Coverage metrics and markdown report...")
    compute_all_steps_ec(
        data_file=data_bundle_path,
        extraction_metrics=str(extraction_metrics_path),
        save_path=str(out_p),
        batch_size=batch_size
    )

    print("\n" + "="*75)
    print(f"🎉 Pre-FEM Block Validation Finished Successfully!")
    print(f"   Outputs located in: {out_p}")
    print("="*75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run Pre-FEM Block Validation immediately after distillation.")
    parser.add_argument("--distilled_dir", type=str, required=True, help="Path to distilled directory")
    parser.add_argument("--extracted_dir", type=str, default=None, help="Path to extracted GP directory")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to synthetic or experimental dataset npz")
    parser.add_argument("--save_path", type=str, default=None, help="Output directory (default: <distilled_dir>/block_validation)")
    parser.add_argument("--step", type=int, default=19, help="Load step to evaluate (default: 19)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for JAX evaluation")
    args = parser.parse_args()

    run_block_validation(
        distilled_dir=args.distilled_dir,
        extracted_dir=args.extracted_dir,
        dataset_path=args.dataset_path,
        save_path=args.save_path,
        step=args.step,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
