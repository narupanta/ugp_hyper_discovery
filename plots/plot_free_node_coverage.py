#!/usr/bin/env python3
"""
plots/plot_free_node_coverage.py

Visualizes spatial domain coverage of distilled hyperelastic material models
based on the free node equilibrium residual distribution R = f_int across 1024 parameter samples.
A free node is defined as covered if 0 falls within the 95% credible interval [q_0.025, q_0.975]
for all its free degrees of freedom. Uncovered nodes are highlighted as prominent red dots.

Generates:
1. Domain plot for True Displacement (Step 19)
2. Domain plot for Observed Displacement (Step 19)
3. Side-by-side comparison domain plot (True vs Observed)
4. Multi-panel distribution & domain coverage plots (matching free_node_residuals_block style)
5. Comprehensive step-by-step coverage metrics JSON
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.stats import gaussian_kde

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from core.utils import deformation_gradient_element, transformation_jacobian
from plots.plot_reaction_force_distilled import piola_stress_2d, format_sci, apply_style, save_figure


def parse_args():
    parser = argparse.ArgumentParser(description="Plot free node equilibrium domain coverage")
    parser.add_argument(
        "--data_file",
        type=str,
        default="results/20261001T223130_isihara_0.0001_0.02_1.0_0.5_5_1.0_isotropic_block/1/fem_validation/block/fem_distilled_samples.npz",
        help="Path to fem_distilled_samples.npz"
    )
    parser.add_argument("--step", type=int, default=19, help="Load step to plot (default: 19)")
    parser.add_argument("--save_path", type=str, default=None, help="Directory to save output plots")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for sample evaluation")
    return parser.parse_args()


def compute_residuals_and_coverage(data_file: str, step: int = 19, batch_size: int = 128):
    """
    Computes VFM free nodal force residuals across all 1024 parameter samples for both
    true and observed displacements at the specified step, evaluating 95% coverage.
    """
    data = np.load(data_file, allow_pickle=True)
    coords = jnp.array(data["node_coords"])
    cells = jnp.array(data["cells"])
    params = jnp.array(data["selected_samples"])
    node_type = np.array(data["node_type"])
    loads = np.array(data["loads"])
    schedule_solve = np.array(data["schedule_solve"])
    control_mode = str(data["control_mode"]).lower() if "control_mode" in data else "displacement"
    stress_mode = str(data["stress_mode"]).lower() if "stress_mode" in data else "plane_stress"

    coords_elems = coords[cells]
    J = transformation_jacobian(coords_elems)
    dA = 0.5 * jnp.abs(jnp.linalg.det(J))

    n_nodes = coords.shape[0]
    n_samples = params.shape[0]
    n_steps = loads.shape[0]
    step_eval = min(step, n_steps - 1)

    is_constrained_x = (node_type[:, 1] == 1) | (node_type[:, 3] == 1)
    is_constrained_y = (node_type[:, 2] == 1) | (node_type[:, 4] == 1)
    is_boundary = is_constrained_x | is_constrained_y
    free_nodes_idx = np.where(~is_boundary)[0]
    dirichlet_nodes_idx = np.where(is_boundary)[0]
    free_x_idx = free_nodes_idx
    free_y_idx = free_nodes_idx

    results = {
        "coords": np.array(coords),
        "cells": np.array(cells),
        "step": step_eval,
        "n_nodes": n_nodes,
        "n_samples": n_samples,
        "free_nodes_idx": free_nodes_idx,
        "free_x_idx": free_x_idx,
        "free_y_idx": free_y_idx,
        "dirichlet_nodes_idx": dirichlet_nodes_idx,
        "disp_prescribed": schedule_solve[step_eval],
        "load_applied": loads[step_eval],
    }

    for disp_name, u_data in [("true", data["u_true"]), ("exp", data["u_exp"])]:
        u_step = jnp.array(u_data[step_eval])
        disp_elems = u_step[cells]
        F_cells, dNdX = deformation_gradient_element(coords_elems, disp_elems)

        @jax.jit
        def eval_fint_one(p):
            P_cells = jax.vmap(lambda f: piola_stress_2d(f, p, stress_mode=stress_mode))(F_cells)
            f_elem = jnp.einsum('cij,cnj->cni', P_cells, dNdX) * dA[:, None, None]
            f_int = jnp.zeros((n_nodes, 2), dtype=jnp.float64)
            for a in range(3):
                f_int = f_int.at[cells[:, a]].add(f_elem[:, a])
            return f_int

        f_int_list = []
        for b in range(0, n_samples, batch_size):
            b_p = params[b:min(b + batch_size, n_samples)]
            b_f = jax.vmap(eval_fint_one)(b_p)
            f_int_list.append(np.array(b_f))
        R_all = np.concatenate(f_int_list, axis=0)  # (1024, n_nodes, 2)

        q025 = np.percentile(R_all, 2.5, axis=0)  # (n_nodes, 2)
        q975 = np.percentile(R_all, 97.5, axis=0) # (n_nodes, 2)

        cov_x = (q025[:, 0] <= 0.0) & (q975[:, 0] >= 0.0)
        cov_y = (q025[:, 1] <= 0.0) & (q975[:, 1] >= 0.0)

        free_cov_x = cov_x[free_x_idx]
        free_cov_y = cov_y[free_y_idx]

        node_covered = np.ones(n_nodes, dtype=bool)
        for idx in free_nodes_idx:
            c_x = cov_x[idx] if not is_constrained_x[idx] else True
            c_y = cov_y[idx] if not is_constrained_y[idx] else True
            node_covered[idx] = c_x and c_y

        # Fully constrained Dirichlet nodes are not evaluated for free equilibrium
        node_covered[dirichlet_nodes_idx] = False

        cov_free_nodes = free_nodes_idx[node_covered[free_nodes_idx]]
        uncov_free_nodes = free_nodes_idx[~node_covered[free_nodes_idx]]

        total_free_nodes = len(free_nodes_idx)
        total_free_dofs = len(free_x_idx) + len(free_y_idx)
        cov_dofs_count = int(np.sum(free_cov_x) + np.sum(free_cov_y))

        results[disp_name] = {
            "R_all": R_all,
            "q025": q025,
            "q975": q975,
            "node_covered": node_covered,
            "cov_free_nodes": cov_free_nodes,
            "uncov_free_nodes": uncov_free_nodes,
            "cov_nodal_pct": (len(cov_free_nodes) / total_free_nodes) * 100.0,
            "cov_dof_pct": (cov_dofs_count / total_free_dofs) * 100.0,
            "cov_x_pct": float(np.mean(free_cov_x) * 100.0),
            "cov_y_pct": float(np.mean(free_cov_y) * 100.0),
            "n_cov_nodes": len(cov_free_nodes),
            "n_uncov_nodes": len(uncov_free_nodes),
            "n_cov_dof": cov_dofs_count,
            "n_uncov_dof": total_free_dofs - cov_dofs_count,
            "r_fx": R_all[:, free_x_idx, 0],
            "r_fy": R_all[:, free_y_idx, 1],
            "r_fnorm": np.linalg.norm(R_all[:, free_nodes_idx, :], axis=-1),
            "mean_nodal_r": np.mean(np.linalg.norm(R_all, axis=-1), axis=0),
        }

    return results


def plot_single_domain_coverage(res: dict, disp_key: str, save_path: str):
    """
    Generates a dedicated domain coverage plot highlighting uncovered nodes as red dots.
    Places legend and metrics box outside the domain mesh to prevent any visual obstruction.
    """
    apply_style()
    coords = res["coords"]
    cells = res["cells"]
    step = res["step"]
    sub = res[disp_key]

    triang = tri.Triangulation(coords[:, 0], coords[:, 1], cells)
    is_true = (disp_key == "true")
    disp_label = "True Displacement (No Noise)" if is_true else "Observed Displacement (with DIC Noise)"

    fig, ax = plt.subplots(figsize=(9.2, 7.5))

    # 1. Mesh wireframe
    ax.triplot(triang, color="#d0d0d0", lw=0.6, zorder=1)

    # 2. Constrained Boundary nodes
    dir_idx = res["dirichlet_nodes_idx"]
    if len(dir_idx) > 0:
        ax.scatter(
            coords[dir_idx, 0], coords[dir_idx, 1],
            c="#6c757d", marker="s", s=50, edgecolors="black", lw=0.8,
            label=f"Boundary Nodes (Excluded) ({len(dir_idx)})", zorder=3
        )

    # 3. Covered Free Nodes
    cov_idx = sub["cov_free_nodes"]
    if len(cov_idx) > 0:
        ax.scatter(
            coords[cov_idx, 0], coords[cov_idx, 1],
            c="#28a745", marker="o", s=52, edgecolors="#155724", lw=0.8,
            label=f"Covered Free Nodes ({len(cov_idx)})", zorder=4
        )

    # 4. Uncovered Free Nodes (Red Dots)
    uncov_idx = sub["uncov_free_nodes"]
    ax.scatter(
        coords[uncov_idx, 0], coords[uncov_idx, 1],
        c="#dc3545", marker="o", s=56, edgecolors="#721c24", lw=0.8,
        label=f"Uncovered Free Nodes ({len(uncov_idx)})", zorder=5
    )

    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"Material Coordinate $X$ [m]", fontsize=12)
    ax.set_ylabel(r"Material Coordinate $Y$ [m]", fontsize=12)
    ax.set_title(
        rf"Domain Equilibrium Coverage at Step {step} (BLOCK)" + "\n" + rf"{disp_label}",
        fontsize=13, fontweight="bold", pad=12
    )
    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)

    # Metrics annotation box and legend placed outside on the right side
    metrics_text = (
        r"$\mathbf{Equilibrium\ Coverage\ (95\%\ CI):}$" + "\n"
        r"$\bullet\ \text{Criterion: } 0 \in [q_{0.025}, q_{0.975}]$" + "\n"
        rf"$\bullet\ \text{{Prescribed }} \mathbf{{u}}: [{res['disp_prescribed'][0]:.3f}, {res['disp_prescribed'][1]:.3f}]\ \mathrm{{m}}$" + "\n"
        rf"$\bullet\ \text{{Applied }} \mathbf{{F}}: [{res['load_applied'][0]:.2f}, {res['load_applied'][1]:.2f}]\ \mathrm{{N}}$" + "\n"
        r"---------------------------------" + "\n"
        rf"$\mathbf{{Nodal\ Coverage:}}\ \mathbf{{{sub['cov_nodal_pct']:.1f}\%}}$" + "\n"
        rf"$\quad\rightarrow\ \text{{Covered: }} {sub['n_cov_nodes']}/{len(res['free_nodes_idx'])}$" + "\n"
        rf"$\quad\rightarrow\ \text{{Uncovered: }} {sub['n_uncov_nodes']}/{len(res['free_nodes_idx'])}$" + "\n"
        r"---------------------------------" + "\n"
        rf"$\mathbf{{DoF\ Coverage:}}\ \mathbf{{{sub['cov_dof_pct']:.1f}\%}}$" + "\n"
        rf"$\quad\rightarrow\ X\text{{-DoF Cov: }} {sub['cov_x_pct']:.1f}\%$" + "\n"
        rf"$\quad\rightarrow\ Y\text{{-DoF Cov: }} {sub['cov_y_pct']:.1f}\%$"
    )

    ax.legend(
        bbox_to_anchor=(1.04, 1.0), loc="upper left",
        fontsize=9.5, framealpha=0.92, facecolor="white", edgecolor="#cccccc"
    )
    ax.text(
        1.04, 0.72, metrics_text, transform=ax.transAxes, verticalalignment="top",
        fontsize=9.0, bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", alpha=0.95, edgecolor="#cccccc")
    )

    plt.tight_layout()
    out_prefix = os.path.join(save_path, f"free_node_coverage_step{step}_{disp_key}")
    save_figure(fig, f"{out_prefix}.pdf", make_png=True)
    plt.close(fig)
    print(f"✅ Saved single domain coverage plot: {out_prefix}.png")


def plot_comparison_domain_coverage(res: dict, save_path: str):
    """
    Generates side-by-side comparison of domain coverage: True vs Observed displacement.
    """
    apply_style()
    coords = res["coords"]
    cells = res["cells"]
    step = res["step"]
    triang = tri.Triangulation(coords[:, 0], coords[:, 1], cells)
    dir_idx = res["dirichlet_nodes_idx"]

    fig, axes = plt.subplots(1, 2, figsize=(16.0, 9.0))

    configs = [
        ("true", axes[0], "True Displacement (No Noise)"),
        ("exp", axes[1], "Observed Displacement (with DIC Noise)")
    ]

    for key, ax, title_suffix in configs:
        sub = res[key]
        ax.triplot(triang, color="#d0d0d0", lw=0.6, zorder=1)

        if len(dir_idx) > 0:
            ax.scatter(
                coords[dir_idx, 0], coords[dir_idx, 1],
                c="#6c757d", marker="s", s=50, edgecolors="black", lw=0.8,
                label=f"Boundary Nodes (Excluded) ({len(dir_idx)})", zorder=3
            )

        cov_idx = sub["cov_free_nodes"]
        if len(cov_idx) > 0:
            ax.scatter(
                coords[cov_idx, 0], coords[cov_idx, 1],
                c="#28a745", marker="o", s=48, edgecolors="#155724", lw=0.8,
                label=f"Covered Nodes ({len(cov_idx)})", zorder=4
            )

        uncov_idx = sub["uncov_free_nodes"]
        ax.scatter(
            coords[uncov_idx, 0], coords[uncov_idx, 1],
            c="#dc3545", marker="o", s=52, edgecolors="#721c24", lw=0.8,
            label=f"Uncovered Nodes ({len(uncov_idx)})", zorder=5
        )

        ax.set_aspect("equal")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(r"Material Coordinate $X$ [m]", fontsize=12)
        ax.set_ylabel(r"Material Coordinate $Y$ [m]", fontsize=12)
        ax.set_title(title_suffix, fontsize=13, fontweight="bold", pad=12)
        ax.grid(True, linestyle="--", alpha=0.3, zorder=0)

        # Place summary badge at bottom
        box_text = (
            rf"$\mathbf{{Nodal\ Coverage:}}\ \mathbf{{{sub['cov_nodal_pct']:.1f}\%}}\ ({sub['n_cov_nodes']}/{len(res['free_nodes_idx'])}) \quad \mid \quad \mathbf{{DoF\ Coverage:}}\ \mathbf{{{sub['cov_dof_pct']:.1f}\%}}$" + "\n"
            rf"$X\text{{-DoF: }} {sub['cov_x_pct']:.1f}\% \quad \mid \quad Y\text{{-DoF: }} {sub['cov_y_pct']:.1f}\% \quad \mid \quad \mathbf{{Uncovered:}}\ \mathbf{{{sub['n_uncov_nodes']}}}\ \text{{nodes}}$"
        )
        ax.text(
            0.5, -0.17, box_text, transform=ax.transAxes, verticalalignment="top", horizontalalignment="center",
            fontsize=9.5, bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8f9fa", alpha=0.95, edgecolor="#cccccc")
        )

    # Place a single shared legend at top center
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.94),
        fontsize=10.5, framealpha=0.92, facecolor="white", edgecolor="#cccccc"
    )

    fig.suptitle(
        rf"Free Node Equilibrium Coverage at Final Load Step {step} (BLOCK)",
        fontsize=15, fontweight="bold", y=0.99
    )
    plt.subplots_adjust(top=0.86, bottom=0.18, wspace=0.25)
    out_prefix = os.path.join(save_path, f"free_node_coverage_step{step}_comparison")
    save_figure(fig, f"{out_prefix}.pdf", make_png=True)
    plt.close(fig)
    print(f"✅ Saved comparison domain coverage plot: {out_prefix}.png")


def plot_four_panel_residuals_coverage(res: dict, disp_key: str, save_path: str):
    """
    Generates a 4-panel figure matching free_node_residuals_block.png:
    Panel 0: Rx distribution
    Panel 1: Ry distribution
    Panel 2: ||R|| distribution
    Panel 3: Domain coverage map with red dots on uncovered nodes
    """
    apply_style()
    coords = res["coords"]
    cells = res["cells"]
    step = res["step"]
    sub = res[disp_key]
    triang = tri.Triangulation(coords[:, 0], coords[:, 1], cells)
    dir_idx = res["dirichlet_nodes_idx"]

    r_fx = sub["r_fx"].flatten()
    r_fy = sub["r_fy"].flatten()
    r_fnorm = sub["r_fnorm"].flatten()

    is_true = (disp_key == "true")
    disp_title = "True Displacement" if is_true else "Observed Displacement"

    fig, axes = plt.subplots(1, 4, figsize=(23.5, 5.0))

    # --- 1. Rx Distribution ---
    ax0 = axes[0]
    mu_x = np.mean(r_fx)
    std_x = np.std(r_fx)
    q025_x, q975_x = np.percentile(r_fx, [2.5, 97.5])
    kde_x = gaussian_kde(r_fx)
    gx = np.linspace(np.percentile(r_fx, 0.5), np.percentile(r_fx, 99.5), 300)
    dens_x = kde_x(gx)

    ax0.hist(r_fx, bins=45, density=True, alpha=0.32, color="#1f77b4", edgecolor="black", lw=0.5)
    ax0.plot(gx, dens_x, color="#1f77b4", lw=2.2, label="Distilled Posterior")
    ax0.axvline(0.0, color="red", linestyle="--", lw=1.6, label="Equilibrium ($R=0$)")
    ax0.axvline(mu_x, color="navy", linestyle="-", lw=1.5, label=rf"Mean: {format_sci(mu_x)}")
    ax0.axvspan(q025_x, q975_x, color="#1f77b4", alpha=0.14, label=r"Distilled 95% CI")
    ax0.set_xlabel(r"Free Node Residual $R_x$", fontsize=12)
    ax0.set_ylabel("Probability Density", fontsize=12)
    ax0.set_title(r"$R_x$ Distribution", fontsize=13, fontweight="bold")
    ax0.grid(True, alpha=0.25, linestyle="--")
    ax0.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    box_x = (
        r"$\mathbf{Distilled\ R_x:}$" + "\n"
        rf"$\mu = {format_sci(mu_x)}$" + "\n"
        rf"$\sigma = {format_sci(std_x)}$" + "\n"
        rf"$95\%\ \mathrm{{CI}}: [{format_sci(q025_x)}, {format_sci(q975_x)}]$" + "\n"
        rf"DoF Cov: {sub['cov_x_pct']:.1f}%"
    )
    ax0.text(0.05, 0.95, box_x, transform=ax0.transAxes, verticalalignment="top",
             fontsize=8.5, bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.88, edgecolor="#cccccc"))

    # --- 2. Ry Distribution ---
    ax1 = axes[1]
    mu_y = np.mean(r_fy)
    std_y = np.std(r_fy)
    q025_y, q975_y = np.percentile(r_fy, [2.5, 97.5])
    kde_y = gaussian_kde(r_fy)
    gy = np.linspace(np.percentile(r_fy, 0.5), np.percentile(r_fy, 99.5), 300)
    dens_y = kde_y(gy)

    ax1.hist(r_fy, bins=45, density=True, alpha=0.32, color="#2ca02c", edgecolor="black", lw=0.5)
    ax1.plot(gy, dens_y, color="#2ca02c", lw=2.2, label="Distilled Posterior")
    ax1.axvline(0.0, color="red", linestyle="--", lw=1.6, label="Equilibrium ($R=0$)")
    ax1.axvline(mu_y, color="darkgreen", linestyle="-", lw=1.5, label=rf"Mean: {format_sci(mu_y)}")
    ax1.axvspan(q025_y, q975_y, color="#2ca02c", alpha=0.14, label=r"Distilled 95% CI")
    ax1.set_xlabel(r"Free Node Residual $R_y$", fontsize=12)
    ax1.set_ylabel("Probability Density", fontsize=12)
    ax1.set_title(r"$R_y$ Distribution", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.25, linestyle="--")
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    box_y = (
        r"$\mathbf{Distilled\ R_y:}$" + "\n"
        rf"$\mu = {format_sci(mu_y)}$" + "\n"
        rf"$\sigma = {format_sci(std_y)}$" + "\n"
        rf"$95\%\ \mathrm{{CI}}: [{format_sci(q025_y)}, {format_sci(q975_y)}]$" + "\n"
        rf"DoF Cov: {sub['cov_y_pct']:.1f}%"
    )
    ax1.text(0.05, 0.95, box_y, transform=ax1.transAxes, verticalalignment="top",
             fontsize=8.5, bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.88, edgecolor="#cccccc"))

    # --- 3. Residual Norm ||R|| Distribution ---
    ax2 = axes[2]
    mu_norm = np.mean(r_fnorm)
    std_norm = np.std(r_fnorm)
    q95_norm = np.percentile(r_fnorm, 95.0)
    kde_norm = gaussian_kde(r_fnorm)
    gn = np.linspace(0, np.percentile(r_fnorm, 99.5), 300)
    dens_n = kde_norm(gn)

    ax2.hist(r_fnorm, bins=45, density=True, alpha=0.32, color="#ff7f0e", edgecolor="black", lw=0.5)
    ax2.plot(gn, dens_n, color="#ff7f0e", lw=2.2, label="Distilled Posterior")
    ax2.axvline(mu_norm, color="darkred", linestyle="-", lw=1.5, label=rf"Mean: {format_sci(mu_norm)}")
    ax2.axvline(q95_norm, color="#ff7f0e", linestyle=":", lw=1.5, label=rf"95th %ile: {format_sci(q95_norm)}")
    ax2.set_xlabel(r"Residual Norm $\|\mathbf{R}_{\mathrm{free}}\|$", fontsize=12)
    ax2.set_ylabel("Probability Density", fontsize=12)
    ax2.set_title(r"$\Vert\mathbf{R}\Vert$ Distribution", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.25, linestyle="--")
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    box_n = (
        r"$\mathbf{Distilled\ \Vert R \Vert:}$" + "\n"
        rf"$\mathrm{{Mean}} = {format_sci(mu_norm)}$" + "\n"
        rf"$\sigma = {format_sci(std_norm)}$" + "\n"
        rf"$\mathrm{{95th\ \%ile}} = {format_sci(q95_norm)}$"
    )
    ax2.text(0.05, 0.95, box_n, transform=ax2.transAxes, verticalalignment="top",
             fontsize=8.5, bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.88, edgecolor="#cccccc"))

    # --- 4. Domain Coverage Plot with Red Dots on Uncovered Nodes ---
    ax3 = axes[3]
    ax3.triplot(triang, color="#d8d8d8", lw=0.6, zorder=1)

    if len(dir_idx) > 0:
        ax3.scatter(
            coords[dir_idx, 0], coords[dir_idx, 1],
            c="#7f7f7f", marker="s", s=28, edgecolors="black", lw=0.5,
            label=f"Boundary (Excluded) ({len(dir_idx)})", zorder=3
        )

    cov_idx = sub["cov_free_nodes"]
    if len(cov_idx) > 0:
        ax3.scatter(
            coords[cov_idx, 0], coords[cov_idx, 1],
            c="#2ca02c", marker="o", s=38, edgecolors="#1b5e20", lw=0.6,
            label=f"Covered ({len(cov_idx)})", zorder=4
        )

    uncov_idx = sub["uncov_free_nodes"]
    ax3.scatter(
        coords[uncov_idx, 0], coords[uncov_idx, 1],
        c="#d62728", marker="o", s=44, edgecolors="#7f0000", lw=0.7,
        label=f"Uncovered ({len(uncov_idx)})", zorder=5
    )

    ax3.set_aspect("equal")
    ax3.set_xlim(-0.05, 1.05)
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_xlabel(r"X [m]", fontsize=12)
    ax3.set_ylabel(r"Y [m]", fontsize=12)
    ax3.set_title(rf"Domain Coverage ({sub['cov_nodal_pct']:.1f}\%)", fontsize=13, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.25, zorder=0)

    # Place legend neatly below or in upper center with tight spacing
    ax3.legend(loc="lower right", fontsize=8.0, framealpha=0.92, facecolor="white", edgecolor="#cccccc")

    cov_box = (
        rf"$\mathbf{{Nodal:}}\ {sub['cov_nodal_pct']:.1f}\%\ ({sub['n_cov_nodes']}/{len(res['free_nodes_idx'])}) $" + "\n"
        rf"$\mathbf{{DoF:}}\ {sub['cov_dof_pct']:.1f}\% \mid \mathbf{{Uncov:}}\ {sub['n_uncov_nodes']}$"
    )
    ax3.text(0.04, 0.96, cov_box, transform=ax3.transAxes, verticalalignment="top",
             fontsize=8.0, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.92, edgecolor="#cccccc"))

    fig.suptitle(
        rf"Free Node Residuals & Domain Coverage at Load Step {step} (BLOCK) — {disp_title}",
        fontsize=15, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    out_prefix = os.path.join(save_path, f"free_node_residuals_coverage_step{step}_{disp_key}")
    save_figure(fig, f"{out_prefix}.pdf", make_png=True)
    plt.close(fig)
    print(f"✅ Saved 4-panel residual coverage plot: {out_prefix}.png")


def compute_and_save_all_steps_metrics(data_file: str, save_path: str, batch_size: int = 128):
    """
    Computes equilibrium coverage metrics across all load steps for both true and
    observed displacements and exports a comprehensive JSON file.
    """
    data = np.load(data_file, allow_pickle=True)
    coords = jnp.array(data["node_coords"])
    cells = jnp.array(data["cells"])
    params = jnp.array(data["selected_samples"])
    node_type = np.array(data["node_type"])
    loads = np.array(data["loads"])
    schedule_solve = np.array(data["schedule_solve"])
    stress_mode = str(data["stress_mode"]).lower() if "stress_mode" in data else "plane_stress"

    coords_elems = coords[cells]
    J = transformation_jacobian(coords_elems)
    dA = 0.5 * jnp.abs(jnp.linalg.det(J))
    n_nodes = coords.shape[0]
    n_samples = params.shape[0]
    n_steps = loads.shape[0]

    is_constrained_x = (node_type[:, 1] == 1) | (node_type[:, 3] == 1)
    is_constrained_y = (node_type[:, 2] == 1) | (node_type[:, 4] == 1)
    is_boundary = is_constrained_x | is_constrained_y
    free_nodes_idx = np.where(~is_boundary)[0]
    dirichlet_nodes_idx = np.where(is_boundary)[0]
    free_x_idx = free_nodes_idx
    free_y_idx = free_nodes_idx
    total_free_nodes = len(free_nodes_idx)
    total_free_dofs = len(free_x_idx) + len(free_y_idx)

    all_steps = []
    print(f"📊 Evaluating all {n_steps} load steps for metrics summary...")

    for s in range(n_steps):
        step_dict = {
            "step": s,
            "prescribed_disp": [float(schedule_solve[s, 0]), float(schedule_solve[s, 1])],
            "applied_load": [float(loads[s, 0]), float(loads[s, 1])],
        }

        for disp_name, u_arr in [("true", data["u_true"]), ("exp", data["u_exp"])]:
            u_step = jnp.array(u_arr[s])
            disp_elems = u_step[cells]
            F_cells, dNdX = deformation_gradient_element(coords_elems, disp_elems)

            @jax.jit
            def eval_fint(p):
                P_cells = jax.vmap(lambda f: piola_stress_2d(f, p, stress_mode=stress_mode))(F_cells)
                f_elem = jnp.einsum('cij,cnj->cni', P_cells, dNdX) * dA[:, None, None]
                f_int = jnp.zeros((n_nodes, 2), dtype=jnp.float64)
                for a in range(3):
                    f_int = f_int.at[cells[:, a]].add(f_elem[:, a])
                return f_int

            f_int_list = []
            for b in range(0, n_samples, batch_size):
                b_p = params[b:min(b + batch_size, n_samples)]
                b_f = jax.vmap(eval_fint)(b_p)
                f_int_list.append(np.array(b_f))
            R_all = np.concatenate(f_int_list, axis=0)

            q025 = np.percentile(R_all, 2.5, axis=0)
            q975 = np.percentile(R_all, 97.5, axis=0)

            cov_x = (q025[:, 0] <= 0.0) & (q975[:, 0] >= 0.0)
            cov_y = (q025[:, 1] <= 0.0) & (q975[:, 1] >= 0.0)

            free_cov_x = cov_x[free_x_idx]
            free_cov_y = cov_y[free_y_idx]

            node_covered = np.ones(n_nodes, dtype=bool)
            for idx in free_nodes_idx:
                c_x = cov_x[idx] if not is_constrained_x[idx] else True
                c_y = cov_y[idx] if not is_constrained_y[idx] else True
                node_covered[idx] = c_x and c_y

            cov_free_count = int(np.sum(node_covered[free_nodes_idx]))
            cov_dof_count = int(np.sum(free_cov_x) + np.sum(free_cov_y))

            step_dict[disp_name] = {
                "cov_nodal_pct": float((cov_free_count / total_free_nodes) * 100.0),
                "cov_dof_pct": float((cov_dof_count / total_free_dofs) * 100.0),
                "cov_x_pct": float(np.mean(free_cov_x) * 100.0),
                "cov_y_pct": float(np.mean(free_cov_y) * 100.0),
                "n_cov_nodes": cov_free_count,
                "n_uncov_nodes": total_free_nodes - cov_free_count,
                "n_cov_dofs": cov_dof_count,
                "n_uncov_dofs": total_free_dofs - cov_dof_count,
            }

        all_steps.append(step_dict)

    out_json = os.path.join(save_path, "free_node_coverage_all_steps.json")
    with open(out_json, "w") as f:
        json.dump(all_steps, f, indent=2)
    print(f"✅ Saved full 20-step coverage JSON: {out_json}")
    return all_steps


def main():
    args = parse_args()
    data_file = args.data_file
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    save_path = args.save_path
    if save_path is None:
        save_path = os.path.dirname(data_file)
    os.makedirs(save_path, exist_ok=True)

    print(f"📊 Computing residuals and coverage for Step {args.step}...")
    res = compute_residuals_and_coverage(data_file, step=args.step, batch_size=args.batch_size)

    # 1. Standalone domain coverage plots
    plot_single_domain_coverage(res, "true", save_path)
    plot_single_domain_coverage(res, "exp", save_path)

    # 2. Comparison domain coverage plot
    plot_comparison_domain_coverage(res, save_path)

    # 3. 4-panel distribution + domain coverage plots
    plot_four_panel_residuals_coverage(res, "true", save_path)
    plot_four_panel_residuals_coverage(res, "exp", save_path)

    # 4. Multi-step evaluation JSON
    compute_and_save_all_steps_metrics(data_file, save_path, batch_size=args.batch_size)

    print("🎉 All domain coverage visualizations and metrics successfully generated!")


if __name__ == "__main__":
    main()
