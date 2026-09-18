#!/usr/bin/env python3
"""
plots/plot_free_node_residuals.py

Computes and visualizes the free node equilibrium residuals (R = f_int - f_ext)
across all distilled material parameter realizations evaluated on the observed
displacement field (u_obs / u_exp), overlaid with the learned Gaussian Process (GP)
free node residual distribution and learned likelihood variance (sigma_free).
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.stats import gaussian_kde, norm
import json

import jax
import jax.numpy as jnp
import jax.random as jr
jax.config.update("jax_enable_x64", True)

from core.dataclass import GPRawParams
from core.model import SparseHyperelasticityGP
from core.utils import deformation_gradient_element, transformation_jacobian
from core.loss_function import neumann_cell_force
from plots.plot_reaction_force_distilled import piola_stress_2d, format_sci, apply_style, save_figure


def parse_args():
    parser = argparse.ArgumentParser(description="Plot free node residuals for distilled material models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fem_validation/{block,holes} or seed directory")
    parser.add_argument("--gp_dir", type=str, default=None, help="Path to extracted GP directory")
    parser.add_argument("--step", type=int, default=16, help="Load step index to evaluate (default: 16)")
    parser.add_argument("--save_path", type=str, default=None, help="Output directory for plots")
    parser.add_argument("--make_png", action="store_true", default=True, help="Save PNG along with PDF")
    return parser.parse_args()


def compute_distilled_free_residuals(data_file: str, step: int = 16, batch_size: int = 64):
    """
    Computes free node residuals R = f_int - f_neu across all distilled parameter samples
    evaluated on u_obs at the specified load step.
    """
    data = np.load(data_file, allow_pickle=True)
    coords = jnp.array(data["node_coords"])
    cells = jnp.array(data["cells"])
    loads = jnp.array(data["loads"])
    params = jnp.array(data["selected_samples"])
    node_type = jnp.array(data["node_type"])

    if "u_exp" in data:
        u_obs = jnp.array(data["u_exp"])
    elif "u_true" in data:
        u_obs = jnp.array(data["u_true"])
    elif "u_pred" in data:
        u_obs = jnp.mean(jnp.array(data["u_pred"]), axis=0)
    else:
        raise KeyError(f"No displacement data found in {data_file}")

    n_samples = params.shape[0]
    n_steps, n_nodes, _ = u_obs.shape
    step_eval = min(step, n_steps - 1)

    coords_elems = coords[cells]
    J = transformation_jacobian(coords_elems)
    dA = 0.5 * jnp.abs(jnp.linalg.det(J))

    is_fix_x = (node_type[:, 1] == 1)
    is_fix_y = (node_type[:, 2] == 1)

    free_x_idx = np.where(~np.array(is_fix_x))[0]
    free_y_idx = np.where(~np.array(is_fix_y))[0]
    free_nodes_idx = np.where((~np.array(is_fix_x)) | (~np.array(is_fix_y)))[0]

    u_step = u_obs[step_eval]
    load_step = loads[step_eval]
    disp_elems = u_step[cells]
    F_cells, dNdX = deformation_gradient_element(coords_elems, disp_elems)

    # Compute external Neumann force on nodes for this step
    onehot_types_el = node_type[cells]
    f_neu_cells = jax.vmap(neumann_cell_force, in_axes=(0, 0, None, None))(
        coords_elems, onehot_types_el, load_step[0], load_step[1]
    )
    f_neu_nodes = jnp.zeros((n_nodes, 2), dtype=jnp.float64)
    for a in range(3):
        f_neu_nodes = f_neu_nodes.at[cells[:, a]].add(f_neu_cells[:, a])

    @jax.jit
    def eval_one_sample(p):
        P_cells = jax.vmap(piola_stress_2d, in_axes=(0, None))(F_cells, p)
        f_elem = jnp.einsum("cij,cnj->cni", P_cells, dNdX) * dA[:, None, None]
        f_int = jnp.zeros((n_nodes, 2), dtype=jnp.float64)
        for a in range(3):
            f_int = f_int.at[cells[:, a]].add(f_elem[:, a])
        
        # Residual at all nodes: R = f_int - f_neu
        R = f_int - f_neu_nodes
        r_free_x = R[free_x_idx, 0]
        r_free_y = R[free_y_idx, 1]
        r_free_norm = jnp.sqrt(R[free_nodes_idx, 0]**2 + R[free_nodes_idx, 1]**2)
        return r_free_x, r_free_y, r_free_norm, R

    r_fx_list, r_fy_list, r_fnorm_list, r_all_list = [], [], [], []
    for b in range(0, n_samples, batch_size):
        b_end = min(b + batch_size, n_samples)
        b_fx, b_fy, b_fnorm, b_r_all = jax.vmap(eval_one_sample)(params[b:b_end])
        r_fx_list.append(np.array(b_fx))
        r_fy_list.append(np.array(b_fy))
        r_fnorm_list.append(np.array(b_fnorm))
        r_all_list.append(np.array(b_r_all))

    r_fx = np.concatenate(r_fx_list, axis=0)       # (n_samples, n_free_x)
    r_fy = np.concatenate(r_fy_list, axis=0)       # (n_samples, n_free_y)
    r_fnorm = np.concatenate(r_fnorm_list, axis=0) # (n_samples, n_free_nodes)
    r_all_nodes = np.concatenate(r_all_list, axis=0) # (n_samples, n_nodes, 2)

    return {
        "r_fx": r_fx,
        "r_fy": r_fy,
        "r_fnorm": r_fnorm,
        "r_all_nodes": r_all_nodes,
        "coords": np.array(coords),
        "cells": np.array(cells),
        "free_x_idx": free_x_idx,
        "free_y_idx": free_y_idx,
        "free_nodes_idx": free_nodes_idx,
        "n_samples": n_samples,
        "n_free_nodes": len(free_nodes_idx),
        "step": step_eval,
        "load": np.array(load_step),
    }


def compute_gp_free_residuals(data_file: str, gp_dir: str = None, step: int = 16, n_gp_samples: int = 1024, batch_size: int = 64):
    """
    Computes free node residuals for the learned Gaussian Process model evaluated on u_obs.
    """
    if gp_dir is None:
        p = Path(data_file).resolve()
        candidates = []
        for ancestor in [p.parent, p.parent.parent, p.parent.parent.parent, p.parent.parent.parent.parent]:
            candidates.append(ancestor / "extracted")
            candidates.append(ancestor / "extracted_model")
            dist_src = ancestor / "distilled" / "source_extraction_dir.txt"
            if dist_src.exists():
                with open(dist_src, "r") as f:
                    candidates.append(Path(f.read().strip()))
            dev_dist_src = ancestor / "distilled" / "dev_source_extraction_dir.txt"
            if dev_dist_src.exists():
                with open(dev_dist_src, "r") as f:
                    candidates.append(Path(f.read().strip()))
        for c in candidates:
            if c.exists() and (c / "best_params.npy").exists():
                gp_dir = str(c)
                break

    if gp_dir is None or not os.path.exists(os.path.join(gp_dir, "best_params.npy")):
        print(f"[WARN] GP model directory not found for {data_file}. Skipping GP comparison.")
        return None

    try:
        data = np.load(data_file, allow_pickle=True)
        coords = jnp.array(data["node_coords"])
        cells = jnp.array(data["cells"])
        node_type = np.array(data["node_type"])
        loads = jnp.array(data["loads"])
        u_obs = jnp.array(data["u_exp"] if "u_exp" in data else data["u_true"])

        n_steps, n_nodes, _ = u_obs.shape
        step_eval = min(step, n_steps - 1)

        best_params_dict = np.load(os.path.join(gp_dir, "best_params.npy"), allow_pickle=True).item()
        valid_fields = set(GPRawParams._fields)
        gp_params = GPRawParams(**{k: v for k, v in best_params_dict.items() if k in valid_fields})
        I_z = jnp.load(os.path.join(gp_dir, "I_z.npy"))

        dev_z = I_z[:, :2]
        vol_z = I_z[:, 2:3]
        min_dev = jnp.min(dev_z, axis=0)
        min_vol = jnp.min(vol_z, axis=0)
        max_dev = jnp.max(dev_z, axis=0)
        max_vol = jnp.max(vol_z, axis=0)

        metadata_path = os.path.join(gp_dir, "metadata.json")
        cov_mode = "diag"
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                cov_mode = json.load(f).get("covariance_mode", "diag")

        gp = SparseHyperelasticityGP(
            gp_params, I_z, min_dev, min_vol, max_dev, max_vol,
            beta=1.0, covariance_mode=cov_mode
        )

        is_fix_x = (node_type[:, 1] == 1)
        is_fix_y = (node_type[:, 2] == 1)
        free_x_idx = np.where(~is_fix_x)[0]
        free_y_idx = np.where(~is_fix_y)[0]
        free_nodes_idx = np.where((~is_fix_x) | (~is_fix_y))[0]

        coords_elems = coords[cells]
        J = transformation_jacobian(coords_elems)
        dA = 0.5 * jnp.abs(jnp.linalg.det(J))
        disp_elems = u_obs[step_eval][cells]
        F_cells, dNdX = deformation_gradient_element(coords_elems, disp_elems)
        F_3d = jnp.eye(3, dtype=jnp.float64)[None, :, :].repeat(F_cells.shape[0], axis=0).at[:, :2, :2].set(F_cells)

        onehot_types_el = node_type[cells]
        load_step = loads[step_eval]
        f_neu_cells = jax.vmap(neumann_cell_force, in_axes=(0, 0, None, None))(
            coords_elems, onehot_types_el, load_step[0], load_step[1]
        )
        f_neu_nodes = jnp.zeros((n_nodes, 2), dtype=jnp.float64)
        for a in range(3):
            f_neu_nodes = f_neu_nodes.at[cells[:, a]].add(f_neu_cells[:, a])

        keys = jr.split(jr.PRNGKey(42), n_gp_samples)

        def eval_gp_sample(k):
            P_3d = jax.vmap(lambda f: gp.piola(f, k))(F_3d)
            P_2d = P_3d[:, :2, :2]
            f_elem = jnp.einsum("cij,cnj->cni", P_2d, dNdX) * dA[:, None, None]
            f_int = jnp.zeros((n_nodes, 2), dtype=jnp.float64)
            for a in range(3):
                f_int = f_int.at[cells[:, a]].add(f_elem[:, a])
            R = f_int - f_neu_nodes
            return R[free_x_idx, 0], R[free_y_idx, 1], jnp.sqrt(R[free_nodes_idx, 0]**2 + R[free_nodes_idx, 1]**2)

        r_x_list, r_y_list, r_norm_list = [], [], []
        for i in range(0, n_gp_samples, batch_size):
            b_keys = keys[i:i + batch_size]
            bx, by, bnorm = jax.vmap(eval_gp_sample)(b_keys)
            r_x_list.append(np.array(bx))
            r_y_list.append(np.array(by))
            r_norm_list.append(np.array(bnorm))

        r_x_gp = np.concatenate(r_x_list, axis=0)
        r_y_gp = np.concatenate(r_y_list, axis=0)
        r_norm_gp = np.concatenate(r_norm_list, axis=0)

        sigma_free_x = float(gp.params.sigma_free_x)
        sigma_free_y = float(gp.params.sigma_free_y)

        return {
            "r_fx_gp": r_x_gp,
            "r_fy_gp": r_y_gp,
            "r_fnorm_gp": r_norm_gp,
            "sigma_free_x": sigma_free_x,
            "sigma_free_y": sigma_free_y,
            "n_gp_samples": n_gp_samples,
        }
    except Exception as e:
        print(f"[WARN] Failed computing GP free residuals: {e}")
        return None


def plot_free_node_residuals(res: dict, geom: str, save_path: str, res_gp: dict = None, make_png: bool = True):
    """
    Plots the probability distributions (Rx, Ry, ||R||) and spatial contour
    of the free node residuals across parameter samples with GP overlay.
    """
    apply_style()
    r_fx = res["r_fx"].flatten()       # (n_samples * n_free_x,)
    r_fy = res["r_fy"].flatten()       # (n_samples * n_free_y,)
    r_fnorm = res["r_fnorm"].flatten() # (n_samples * n_free_nodes,)
    
    # Nodal mean norm across samples
    mean_nodal_r = np.mean(np.linalg.norm(res["r_all_nodes"], axis=-1), axis=0) # (n_nodes,)
    
    step = res["step"]
    geom_title = geom.upper()

    fig, axes = plt.subplots(1, 4, figsize=(23, 4.9))

    # --- 1. Residual X Distribution ---
    ax0 = axes[0]
    mu_x = np.mean(r_fx)
    std_x = np.std(r_fx)
    q025_x, q975_x = np.percentile(r_fx, [2.5, 97.5])
    
    kde_x = gaussian_kde(r_fx)
    gx = np.linspace(np.percentile(r_fx, 0.5), np.percentile(r_fx, 99.5), 300)
    dens_x = kde_x(gx)
    
    ax0.hist(r_fx, bins=45, density=True, alpha=0.32, color='#1f77b4', edgecolor='black', lw=0.5)
    ax0.plot(gx, dens_x, color='#1f77b4', lw=2.2, label="Distilled Posterior")
    ax0.axvline(0.0, color='red', linestyle='--', lw=1.6, label="Equilibrium ($R=0$)")
    ax0.axvline(mu_x, color='navy', linestyle='-', lw=1.5, label=rf"Mean: {format_sci(mu_x)}")
    ax0.axvspan(q025_x, q975_x, color='#1f77b4', alpha=0.14, label=r"Distilled 95% CI")
    
    # GP Overlay if available
    if res_gp is not None:
        r_fx_gp = res_gp["r_fx_gp"].flatten()
        kde_x_gp = gaussian_kde(r_fx_gp)
        dens_x_gp = kde_x_gp(gx)
        ax0.plot(gx, dens_x_gp, color='#555555', linestyle='--', lw=2.0, label="Learned GP Posterior")
        sig_fx = res_gp.get("sigma_free_x", None)
        if sig_fx is not None:
            # Learned likelihood Gaussian
            norm_fx = norm.pdf(gx, loc=0.0, scale=sig_fx)
            ax0.plot(gx, norm_fx, color='#888888', linestyle=':', lw=1.6, label=rf"Likelihood $\sigma_{{\mathrm{{free}}, x}}$ ({format_sci(sig_fx)})")

    ax0.set_xlabel(r"Free Node Residual $R_x$", fontsize=12)
    ax0.set_ylabel("Probability Density", fontsize=12)
    ax0.set_title(rf"$R_x$ Distribution ({geom_title})", fontsize=13, fontweight='bold')
    ax0.grid(True, alpha=0.25, linestyle='--')
    ax0.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
    
    box_x = (
        rf"$\mathbf{{Distilled \ R_x:}}$" + "\n"
        rf"$\mu = {format_sci(mu_x)}$" + "\n"
        rf"$\sigma = {format_sci(std_x)}$" + "\n"
        rf"$95\%\ \mathrm{{CI}}: [{format_sci(q025_x)}, {format_sci(q975_x)}]$"
    )
    ax0.text(0.05, 0.95, box_x, transform=ax0.transAxes, verticalalignment='top',
             fontsize=8.8, bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.88, edgecolor='#cccccc'))

    # --- 2. Residual Y Distribution ---
    ax1 = axes[1]
    mu_y = np.mean(r_fy)
    std_y = np.std(r_fy)
    q025_y, q975_y = np.percentile(r_fy, [2.5, 97.5])
    
    kde_y = gaussian_kde(r_fy)
    gy = np.linspace(np.percentile(r_fy, 0.5), np.percentile(r_fy, 99.5), 300)
    dens_y = kde_y(gy)
    
    ax1.hist(r_fy, bins=45, density=True, alpha=0.32, color='#2ca02c', edgecolor='black', lw=0.5)
    ax1.plot(gy, dens_y, color='#2ca02c', lw=2.2, label="Distilled Posterior")
    ax1.axvline(0.0, color='red', linestyle='--', lw=1.6, label="Equilibrium ($R=0$)")
    ax1.axvline(mu_y, color='darkgreen', linestyle='-', lw=1.5, label=rf"Mean: {format_sci(mu_y)}")
    ax1.axvspan(q025_y, q975_y, color='#2ca02c', alpha=0.14, label=r"Distilled 95% CI")
    
    # GP Overlay if available
    if res_gp is not None:
        r_fy_gp = res_gp["r_fy_gp"].flatten()
        kde_y_gp = gaussian_kde(r_fy_gp)
        dens_y_gp = kde_y_gp(gy)
        ax1.plot(gy, dens_y_gp, color='#555555', linestyle='--', lw=2.0, label="Learned GP Posterior")
        sig_fy = res_gp.get("sigma_free_y", None)
        if sig_fy is not None:
            norm_fy = norm.pdf(gy, loc=0.0, scale=sig_fy)
            ax1.plot(gy, norm_fy, color='#888888', linestyle=':', lw=1.6, label=rf"Likelihood $\sigma_{{\mathrm{{free}}, y}}$ ({format_sci(sig_fy)})")

    ax1.set_xlabel(r"Free Node Residual $R_y$", fontsize=12)
    ax1.set_ylabel("Probability Density", fontsize=12)
    ax1.set_title(rf"$R_y$ Distribution ({geom_title})", fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.25, linestyle='--')
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
    
    box_y = (
        rf"$\mathbf{{Distilled \ R_y:}}$" + "\n"
        rf"$\mu = {format_sci(mu_y)}$" + "\n"
        rf"$\sigma = {format_sci(std_y)}$" + "\n"
        rf"$95\%\ \mathrm{{CI}}: [{format_sci(q025_y)}, {format_sci(q975_y)}]$"
    )
    ax1.text(0.05, 0.95, box_y, transform=ax1.transAxes, verticalalignment='top',
             fontsize=8.8, bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.88, edgecolor='#cccccc'))

    # --- 3. Residual Norm ||R|| Distribution ---
    ax2 = axes[2]
    mu_norm = np.mean(r_fnorm)
    std_norm = np.std(r_fnorm)
    q95_norm = np.percentile(r_fnorm, 95.0)
    
    kde_norm = gaussian_kde(r_fnorm)
    gn = np.linspace(0, np.percentile(r_fnorm, 99.5), 300)
    dens_n = kde_norm(gn)
    
    ax2.hist(r_fnorm, bins=45, density=True, alpha=0.32, color='#ff7f0e', edgecolor='black', lw=0.5)
    ax2.plot(gn, dens_n, color='#ff7f0e', lw=2.2, label="Distilled Posterior")
    ax2.axvline(mu_norm, color='darkred', linestyle='-', lw=1.5, label=rf"Mean: {format_sci(mu_norm)}")
    ax2.axvline(q95_norm, color='#ff7f0e', linestyle=':', lw=1.5, label=rf"95th %ile: {format_sci(q95_norm)}")
    
    if res_gp is not None:
        r_fnorm_gp = res_gp["r_fnorm_gp"].flatten()
        kde_norm_gp = gaussian_kde(r_fnorm_gp)
        dens_norm_gp = kde_norm_gp(gn)
        ax2.plot(gn, dens_norm_gp, color='#555555', linestyle='--', lw=2.0, label="Learned GP Posterior")

    ax2.set_xlabel(r"Residual Norm $\|\mathbf{R}_{\mathrm{free}}\|$", fontsize=12)
    ax2.set_ylabel("Probability Density", fontsize=12)
    ax2.set_title(rf"$\Vert\mathbf{{R}}\Vert$ Distribution ({geom_title})", fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.25, linestyle='--')
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
    
    box_n = (
        rf"$\mathbf{{Distilled \ \Vert R \Vert:}}$" + "\n"
        rf"$\mathrm{{Mean}} = {format_sci(mu_norm)}$" + "\n"
        rf"$\sigma = {format_sci(std_norm)}$" + "\n"
        rf"$\mathrm{{95th\ \%ile}} = {format_sci(q95_norm)}$"
    )
    ax2.text(0.05, 0.95, box_n, transform=ax2.transAxes, verticalalignment='top',
             fontsize=8.8, bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.88, edgecolor='#cccccc'))

    # --- 4. Spatial Heatmap of Mean Nodal Residuals ---
    ax3 = axes[3]
    coords = res["coords"]
    cells = res["cells"]
    triang = tri.Triangulation(coords[:, 0], coords[:, 1], cells)
    
    tcf = ax3.tripcolor(triang, mean_nodal_r, cmap="inferno", shading="gouraud")
    ax3.set_aspect("equal")
    ax3.set_title(rf"Mean Spatial $\Vert\mathbf{{R}}\Vert$ (Step {step})", fontsize=13, fontweight='bold')
    ax3.set_xlabel("X [m]", fontsize=12)
    ax3.set_ylabel("Y [m]", fontsize=12)
    cbar = fig.colorbar(tcf, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\mathbb{E}_p[\Vert\mathbf{R}_i\Vert]$", fontsize=11)

    fig.suptitle(rf"Free Node Equilibrium Residuals at Load Step {step} ({geom_title})",
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_pdf = os.path.join(save_path, f"free_node_residuals_{geom}.pdf")
    save_figure(fig, out_pdf, make_png=make_png)
    plt.close(fig)

    metrics = {
        "geometry": geom,
        "step": step,
        "mean_residual_x": float(mu_x),
        "std_residual_x": float(std_x),
        "q025_residual_x": float(q025_x),
        "q975_residual_x": float(q975_x),
        "mean_residual_y": float(mu_y),
        "std_residual_y": float(std_y),
        "q025_residual_y": float(q025_y),
        "q975_residual_y": float(q975_y),
        "mean_residual_norm": float(mu_norm),
        "std_residual_norm": float(std_norm),
        "q95_residual_norm": float(q95_norm),
        "n_free_nodes": res["n_free_nodes"],
        "n_samples": res["n_samples"],
    }
    if res_gp is not None:
        metrics["gp_sigma_free_x"] = res_gp.get("sigma_free_x")
        metrics["gp_sigma_free_y"] = res_gp.get("sigma_free_y")
        metrics["gp_mean_residual_x"] = float(np.mean(res_gp["r_fx_gp"]))
        metrics["gp_std_residual_x"] = float(np.std(res_gp["r_fx_gp"]))
        metrics["gp_mean_residual_y"] = float(np.mean(res_gp["r_fy_gp"]))
        metrics["gp_std_residual_y"] = float(np.std(res_gp["r_fy_gp"]))

    with open(os.path.join(save_path, f"free_node_residuals_{geom}.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


def main():
    args = parse_args()
    model_path = Path(args.model_path)

    targets = []
    if (model_path / "fem_distilled_samples.npz").exists():
        geom = "holes" if "holes" in str(model_path).lower() else "block"
        save_p = args.save_path or str(model_path)
        targets.append((str(model_path / "fem_distilled_samples.npz"), geom, save_p))
    else:
        for g in ["block", "holes"]:
            cand = model_path / g / "fem_distilled_samples.npz"
            if cand.exists():
                save_p = args.save_path or str(model_path / g)
                targets.append((str(cand), g, save_p))

    if not targets:
        print(f"Error: Could not locate fem_distilled_samples.npz in {model_path}")
        sys.exit(1)

    exp_plots = Path(model_path).parent / "plots" if (Path(model_path).parent / "plots").exists() else None

    for data_file, geom, save_p in targets:
        print(f"--- Computing Free Node Residuals for {geom.upper()} from {data_file} at Step {args.step} ---")
        res = compute_distilled_free_residuals(data_file, step=args.step)
        res_gp = compute_gp_free_residuals(data_file, gp_dir=args.gp_dir, step=args.step)
        metrics = plot_free_node_residuals(res, geom, save_p, res_gp=res_gp, make_png=args.make_png)
        
        # Also save copy to top-level experiment plots/ directory
        if exp_plots and exp_plots.exists():
            plot_free_node_residuals(res, geom, str(exp_plots), res_gp=res_gp, make_png=args.make_png)

        print(f"✅ Saved {geom.upper()} Free Node Residual Plots:\n   {save_p}/free_node_residuals_{geom}.pdf")
        print(f"   Mean Rx: {format_sci(metrics['mean_residual_x'])}, Mean Ry: {format_sci(metrics['mean_residual_y'])}, Mean ||R||: {format_sci(metrics['mean_residual_norm'])}")
        if res_gp is not None:
            print(f"   GP Learned sigma_free_x: {format_sci(res_gp['sigma_free_x'])}, sigma_free_y: {format_sci(res_gp['sigma_free_y'])}")


if __name__ == "__main__":
    main()
