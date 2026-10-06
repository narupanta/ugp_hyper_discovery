#!/usr/bin/env python3
"""
plots/plot_coverage_with_free_variance.py

Incorporates the learned free residual noise variances (sigma_free_x^2, sigma_free_y^2)
from the UGP training model into the FEM equilibrium coverage test at step 19.

Generates:
1. 4-panel figure WITHOUT free residual variance (Observed displacement)
2. 4-panel figure WITH free residual variance (Observed displacement)
3. Side-by-side 4-panel comparison of domain equilibrium coverage (Without vs With variance)
4. 4-node local residual distributions WITHOUT free residual variance
5. 4-node local residual distributions WITH free residual variance
6. Direct comparison overlay for the 4 selected nodes showing Without vs With variance
7. Corresponding figures for True displacement
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
jax.config.update("jax_enable_x64", True)

from plots.theme import apply_style, save_figure
from plots.plot_reaction_force_distilled import format_sci
from plots.plot_free_node_coverage import compute_residuals_and_coverage


def evaluate_coverage_with_and_without_variance(res, sigma_free_x, sigma_free_y, n_mc_draws=100, seed=42):
    """
    Computes nodal and DoF equilibrium coverage for both:
    1. 'without_var': Epistemic/parameter uncertainty only (R ~ p(R | theta))
    2. 'with_var': Total predictive uncertainty (R ~ p(R | theta) convolved with N(0, sigma_free^2))
    """
    np.random.seed(seed)
    coords = res["coords"]
    n_nodes = res["n_nodes"]
    free_x_idx = res["free_x_idx"]
    free_y_idx = res["free_y_idx"]
    free_nodes_idx = res["free_nodes_idx"]
    dirichlet_nodes_idx = res["dirichlet_nodes_idx"]
    
    is_constrained_x = np.isin(np.arange(n_nodes), free_x_idx, invert=True)
    is_constrained_y = np.isin(np.arange(n_nodes), free_y_idx, invert=True)
    
    out = {}
    
    for disp_name in ["true", "exp"]:
        R_all = res[disp_name]["R_all"] # (1024, n_nodes, 2)
        n_samples = R_all.shape[0]
        
        # --- 1. Without Variance ---
        q025_no = np.percentile(R_all, 2.5, axis=0) # (n_nodes, 2)
        q975_no = np.percentile(R_all, 97.5, axis=0)
        
        cov_x_no = (q025_no[:, 0] <= 0.0) & (q975_no[:, 0] >= 0.0)
        cov_y_no = (q025_no[:, 1] <= 0.0) & (q975_no[:, 1] >= 0.0)
        
        node_cov_no = np.ones(n_nodes, dtype=bool)
        for idx in free_nodes_idx:
            c_x = cov_x_no[idx] if not is_constrained_x[idx] else True
            c_y = cov_y_no[idx] if not is_constrained_y[idx] else True
            node_cov_no[idx] = c_x and c_y
        node_cov_no[dirichlet_nodes_idx] = False
        
        cov_free_nodes_no = free_nodes_idx[node_cov_no[free_nodes_idx]]
        uncov_free_nodes_no = free_nodes_idx[~node_cov_no[free_nodes_idx]]
        
        # --- 2. With Variance (1 predictive noise draw per parameter sample: ancestral sampling) ---
        eps_x = np.random.normal(0.0, sigma_free_x, (n_samples, n_nodes))
        eps_y = np.random.normal(0.0, sigma_free_y, (n_samples, n_nodes))
        
        # Shape: (n_samples, n_nodes)
        R_with_x = R_all[:, :, 0] + eps_x
        R_with_y = R_all[:, :, 1] + eps_y
        
        q025_with = np.zeros((n_nodes, 2))
        q975_with = np.zeros((n_nodes, 2))
        q025_with[:, 0] = np.percentile(R_with_x, 2.5, axis=0)
        q975_with[:, 0] = np.percentile(R_with_x, 97.5, axis=0)
        q025_with[:, 1] = np.percentile(R_with_y, 2.5, axis=0)
        q975_with[:, 1] = np.percentile(R_with_y, 97.5, axis=0)
        
        cov_x_with = (q025_with[:, 0] <= 0.0) & (q975_with[:, 0] >= 0.0)
        cov_y_with = (q025_with[:, 1] <= 0.0) & (q975_with[:, 1] >= 0.0)
        
        node_cov_with = np.ones(n_nodes, dtype=bool)
        for idx in free_nodes_idx:
            c_x = cov_x_with[idx] if not is_constrained_x[idx] else True
            c_y = cov_y_with[idx] if not is_constrained_y[idx] else True
            node_cov_with[idx] = c_x and c_y
        node_cov_with[dirichlet_nodes_idx] = False
        
        cov_free_nodes_with = free_nodes_idx[node_cov_with[free_nodes_idx]]
        uncov_free_nodes_with = free_nodes_idx[~node_cov_with[free_nodes_idx]]
        
        total_free_nodes = len(free_nodes_idx)
        total_free_dofs = len(free_x_idx) + len(free_y_idx)
        
        out[disp_name] = {
            "without_var": {
                "R_all": R_all,
                "q025": q025_no,
                "q975": q975_no,
                "cov_x": cov_x_no,
                "cov_y": cov_y_no,
                "node_covered": node_cov_no,
                "cov_free_nodes": cov_free_nodes_no,
                "uncov_free_nodes": uncov_free_nodes_no,
                "cov_nodal_pct": (len(cov_free_nodes_no) / total_free_nodes) * 100.0,
                "cov_dof_pct": ((np.sum(cov_x_no[free_x_idx]) + np.sum(cov_y_no[free_y_idx])) / total_free_dofs) * 100.0,
                "cov_x_pct": float(np.mean(cov_x_no[free_x_idx]) * 100.0),
                "cov_y_pct": float(np.mean(cov_y_no[free_y_idx]) * 100.0),
                "n_cov_nodes": len(cov_free_nodes_no),
                "n_uncov_nodes": len(uncov_free_nodes_no),
                "r_fx": R_all[:, free_x_idx, 0],
                "r_fy": R_all[:, free_y_idx, 1],
                "r_fnorm": np.linalg.norm(R_all[:, free_nodes_idx, :], axis=-1),
            },
            "with_var": {
                "R_with_x": R_with_x,
                "R_with_y": R_with_y,
                "q025": q025_with,
                "q975": q975_with,
                "cov_x": cov_x_with,
                "cov_y": cov_y_with,
                "node_covered": node_cov_with,
                "cov_free_nodes": cov_free_nodes_with,
                "uncov_free_nodes": uncov_free_nodes_with,
                "cov_nodal_pct": (len(cov_free_nodes_with) / total_free_nodes) * 100.0,
                "cov_dof_pct": ((np.sum(cov_x_with[free_x_idx]) + np.sum(cov_y_with[free_y_idx])) / total_free_dofs) * 100.0,
                "cov_x_pct": float(np.mean(cov_x_with[free_x_idx]) * 100.0),
                "cov_y_pct": float(np.mean(cov_y_with[free_y_idx]) * 100.0),
                "n_cov_nodes": len(cov_free_nodes_with),
                "n_uncov_nodes": len(uncov_free_nodes_with),
                "r_fx": R_with_x[:, free_x_idx],
                "r_fy": R_with_y[:, free_y_idx],
                "r_fnorm": np.sqrt(R_with_x[:, free_nodes_idx]**2 + R_with_y[:, free_nodes_idx]**2),
            }
        }
        
    return out


def plot_four_panel_variant(res, cov_data, disp_key, variant_key, sigma_free_x, sigma_free_y, save_path):
    """
    Plots the 4-panel residual distribution and domain coverage map matching
    free_node_residuals_coverage_step19_exp.png style.
    """
    apply_style()
    coords = res["coords"]
    cells = res["cells"]
    step = res["step"]
    sub = cov_data[disp_key][variant_key]
    triang = tri.Triangulation(coords[:, 0], coords[:, 1], cells)
    dir_idx = res["dirichlet_nodes_idx"]

    r_fx = sub["r_fx"].flatten()
    r_fy = sub["r_fy"].flatten()
    r_fnorm = sub["r_fnorm"].flatten()

    is_true = (disp_key == "true")
    disp_title = "True Displacement" if is_true else "Observed Displacement"
    has_var = (variant_key == "with_var")
    var_title = "WITH Learned Residual Variance" if has_var else "WITHOUT Residual Variance"

    fig, axes = plt.subplots(1, 4, figsize=(23.5, 5.0))

    # --- 1. Rx Distribution ---
    ax0 = axes[0]
    mu_x = float(np.mean(r_fx))
    std_x = float(np.std(r_fx))
    q025_x, q975_x = float(np.percentile(r_fx, 2.5)), float(np.percentile(r_fx, 97.5))
    kde_x = gaussian_kde(np.random.choice(r_fx, size=min(len(r_fx), 5000), replace=False))
    gx = np.linspace(np.percentile(r_fx, 0.5), np.percentile(r_fx, 99.5), 300)
    dens_x = kde_x(gx)

    ax0.hist(r_fx, bins=45, density=True, alpha=0.32, color="#1f77b4", edgecolor="black", lw=0.5)
    ax0.plot(gx, dens_x, color="#1f77b4", lw=2.2, label="Predictive Posterior")
    ax0.axvline(0.0, color="red", linestyle="--", lw=1.6, label="Equilibrium ($R=0$)")
    ax0.axvline(mu_x, color="navy", linestyle="-", lw=1.5, label=rf"Mean: {format_sci(mu_x)}")
    ax0.axvspan(q025_x, q975_x, color="#1f77b4", alpha=0.14, label=r"95% CI")
    ax0.set_xlabel(r"Free Node Residual $R_x$", fontsize=12)
    ax0.set_ylabel("Probability Density", fontsize=12)
    ax0.set_title(r"$R_x$ Distribution", fontsize=13, fontweight="bold")
    ax0.grid(True, alpha=0.25, linestyle="--")
    ax0.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    box_x = (
        (rf"$\mathbf{{R_x\ (With\ \sigma_{{free,x}}):}}$" if has_var else r"$\mathbf{{Distilled\ R_x:}}$") + "\n"
        rf"$\mu = {format_sci(mu_x)}$" + "\n"
        rf"$\sigma = {format_sci(std_x)}$" + "\n"
        rf"$95\%\ \mathrm{{CI}}: [{format_sci(q025_x)}, {format_sci(q975_x)}]$" + "\n"
        rf"DoF Cov: {sub['cov_x_pct']:.1f}%"
    )
    if has_var:
        box_x += "\n" + rf"$\sigma_{{free,x}} = {format_sci(sigma_free_x)}$"
    ax0.text(0.05, 0.95, box_x, transform=ax0.transAxes, verticalalignment="top",
             fontsize=8.5, bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.88, edgecolor="#cccccc"))

    # --- 2. Ry Distribution ---
    ax1 = axes[1]
    mu_y = float(np.mean(r_fy))
    std_y = float(np.std(r_fy))
    q025_y, q975_y = float(np.percentile(r_fy, 2.5)), float(np.percentile(r_fy, 97.5))
    kde_y = gaussian_kde(np.random.choice(r_fy, size=min(len(r_fy), 5000), replace=False))
    gy = np.linspace(np.percentile(r_fy, 0.5), np.percentile(r_fy, 99.5), 300)
    dens_y = kde_y(gy)

    ax1.hist(r_fy, bins=45, density=True, alpha=0.32, color="#2ca02c", edgecolor="black", lw=0.5)
    ax1.plot(gy, dens_y, color="#2ca02c", lw=2.2, label="Predictive Posterior")
    ax1.axvline(0.0, color="red", linestyle="--", lw=1.6, label="Equilibrium ($R=0$)")
    ax1.axvline(mu_y, color="darkgreen", linestyle="-", lw=1.5, label=rf"Mean: {format_sci(mu_y)}")
    ax1.axvspan(q025_y, q975_y, color="#2ca02c", alpha=0.14, label=r"95% CI")
    ax1.set_xlabel(r"Free Node Residual $R_y$", fontsize=12)
    ax1.set_ylabel("Probability Density", fontsize=12)
    ax1.set_title(r"$R_y$ Distribution", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.25, linestyle="--")
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    box_y = (
        (rf"$\mathbf{{R_y\ (With\ \sigma_{{free,y}}):}}$" if has_var else r"$\mathbf{{Distilled\ R_y:}}$") + "\n"
        rf"$\mu = {format_sci(mu_y)}$" + "\n"
        rf"$\sigma = {format_sci(std_y)}$" + "\n"
        rf"$95\%\ \mathrm{{CI}}: [{format_sci(q025_y)}, {format_sci(q975_y)}]$" + "\n"
        rf"DoF Cov: {sub['cov_y_pct']:.1f}%"
    )
    if has_var:
        box_y += "\n" + rf"$\sigma_{{free,y}} = {format_sci(sigma_free_y)}$"
    ax1.text(0.05, 0.95, box_y, transform=ax1.transAxes, verticalalignment="top",
             fontsize=8.5, bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.88, edgecolor="#cccccc"))

    # --- 3. Residual Norm ||R|| Distribution ---
    ax2 = axes[2]
    mu_norm = float(np.mean(r_fnorm))
    std_norm = float(np.std(r_fnorm))
    q95_norm = float(np.percentile(r_fnorm, 95.0))
    kde_norm = gaussian_kde(np.random.choice(r_fnorm, size=min(len(r_fnorm), 5000), replace=False))
    gn = np.linspace(0, np.percentile(r_fnorm, 99.5), 300)
    dens_n = kde_norm(gn)

    ax2.hist(r_fnorm, bins=45, density=True, alpha=0.32, color="#ff7f0e", edgecolor="black", lw=0.5)
    ax2.plot(gn, dens_n, color="#ff7f0e", lw=2.2, label="Predictive Posterior")
    ax2.axvline(mu_norm, color="darkred", linestyle="-", lw=1.5, label=rf"Mean: {format_sci(mu_norm)}")
    ax2.axvline(q95_norm, color="#ff7f0e", linestyle=":", lw=1.5, label=rf"95th %ile: {format_sci(q95_norm)}")
    ax2.set_xlabel(r"Residual Norm $\|\mathbf{R}_{\mathrm{free}}\|$", fontsize=12)
    ax2.set_ylabel("Probability Density", fontsize=12)
    ax2.set_title(r"$\Vert\mathbf{R}\Vert$ Distribution", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.25, linestyle="--")
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    box_n = (
        r"$\mathbf{\Vert R \Vert:}$" + "\n"
        rf"$\mathrm{{Mean}} = {format_sci(mu_norm)}$" + "\n"
        rf"$\sigma = {format_sci(std_norm)}$" + "\n"
        rf"$\mathrm{{95th\ \%ile}} = {format_sci(q95_norm)}$"
    )
    ax2.text(0.05, 0.95, box_n, transform=ax2.transAxes, verticalalignment="top",
             fontsize=8.5, bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.88, edgecolor="#cccccc"))

    # --- 4. Domain Coverage Plot ---
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
    if len(uncov_idx) > 0:
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
    ax3.legend(loc="lower right", fontsize=8.0, framealpha=0.92, facecolor="white", edgecolor="#cccccc")

    cov_box = (
        rf"$\mathbf{{Nodal:}}\ {sub['cov_nodal_pct']:.1f}\%\ ({sub['n_cov_nodes']}/{len(res['free_nodes_idx'])}) $" + "\n"
        rf"$\mathbf{{DoF:}}\ {sub['cov_dof_pct']:.1f}\% \mid \mathbf{{Uncov:}}\ {sub['n_uncov_nodes']}$"
    )
    ax3.text(0.04, 0.96, cov_box, transform=ax3.transAxes, verticalalignment="top",
             fontsize=8.0, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.92, edgecolor="#cccccc"))

    fig.suptitle(
        rf"Free Node Residuals & Domain Coverage at Load Step {step} (BLOCK) — {disp_title}" + "\n" +
        rf"[{var_title}]",
        fontsize=15, fontweight="bold", y=1.03
    )
    plt.tight_layout()
    out_prefix = os.path.join(save_path, f"free_node_residuals_coverage_step{step}_{disp_key}_{variant_key}")
    save_figure(fig, f"{out_prefix}.pdf", make_png=True)
    plt.close(fig)
    print(f"✅ Saved 4-panel plot: {out_prefix}.png")


def plot_side_by_side_residuals_coverage_comparison(res, cov_data, disp_key, save_path):
    """
    Plots a side-by-side comparison of the 4-panel domain coverage:
    Left: WITHOUT free residual variance
    Right: WITH learned free residual variance
    """
    apply_style()
    coords = res["coords"]
    cells = res["cells"]
    step = res["step"]
    triang = tri.Triangulation(coords[:, 0], coords[:, 1], cells)
    dir_idx = res["dirichlet_nodes_idx"]

    sub_no = cov_data[disp_key]["without_var"]
    sub_with = cov_data[disp_key]["with_var"]

    disp_label = "Observed Displacement (with DIC Noise)" if disp_key == "exp" else "True Displacement (No Noise)"

    fig, axes = plt.subplots(1, 2, figsize=(16.0, 8.5))

    variants = [
        (sub_no, axes[0], "WITHOUT Free Residual Variance (Epistemic Only)", "#dc3545"),
        (sub_with, axes[1], r"WITH Learned Free Residual Variance ($\sigma_{\mathrm{free}}^2$)", "#28a745")
    ]

    for sub, ax, title_suffix, accent_color in variants:
        ax.triplot(triang, color="#d8d8d8", lw=0.6, zorder=1)

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
                c="#28a745", marker="o", s=52, edgecolors="#155724", lw=0.8,
                label=f"Covered Nodes ({len(cov_idx)})", zorder=4
            )

        uncov_idx = sub["uncov_free_nodes"]
        if len(uncov_idx) > 0:
            ax.scatter(
                coords[uncov_idx, 0], coords[uncov_idx, 1],
                c="#dc3545", marker="o", s=56, edgecolors="#721c24", lw=0.8,
                label=f"Uncovered Nodes ({len(uncov_idx)})", zorder=5
            )

        ax.set_aspect("equal")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(r"Material Coordinate $X$ [m]", fontsize=12)
        ax.set_ylabel(r"Material Coordinate $Y$ [m]", fontsize=12)
        ax.set_title(title_suffix, fontsize=13, fontweight="bold", pad=12, color="#222222")
        ax.grid(True, linestyle="--", alpha=0.3, zorder=0)

        # Summary box
        box_text = (
            rf"$\mathbf{{Nodal\ Coverage:}}\ \mathbf{{{sub['cov_nodal_pct']:.1f}\%}}\ ({sub['n_cov_nodes']}/{len(res['free_nodes_idx'])}) \quad \mid \quad \mathbf{{DoF\ Coverage:}}\ \mathbf{{{sub['cov_dof_pct']:.1f}\%}}$" + "\n"
            rf"$X\text{{-DoF: }} {sub['cov_x_pct']:.1f}\% \quad \mid \quad Y\text{{-DoF: }} {sub['cov_y_pct']:.1f}\% \quad \mid \quad \mathbf{{Uncovered:}}\ \mathbf{{{sub['n_uncov_nodes']}}}\ \text{{nodes}}$"
        )
        ax.text(
            0.5, -0.16, box_text, transform=ax.transAxes, verticalalignment="top", horizontalalignment="center",
            fontsize=9.5, bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8f9fa", alpha=0.95, edgecolor="#cccccc")
        )

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.94),
        fontsize=10.5, framealpha=0.92, facecolor="white", edgecolor="#cccccc"
    )

    fig.suptitle(
        rf"Domain Equilibrium Coverage Impact of Learned Residual Variance (Step {step})" + "\n" +
        rf"{disp_label}",
        fontsize=15, fontweight="bold", y=0.99
    )
    plt.subplots_adjust(top=0.86, bottom=0.17, wspace=0.22)
    out_prefix = os.path.join(save_path, f"free_node_coverage_step{step}_{disp_key}_variance_comparison")
    save_figure(fig, f"{out_prefix}.pdf", make_png=True)
    plt.close(fig)
    print(f"✅ Saved domain variance comparison plot: {out_prefix}.png")


def plot_local_nodes_variant(res, cov_data, selected_nodes, disp_key, variant_key, sigma_free_x, sigma_free_y, save_path):
    """
    Plots a 2x4 grid of residual distributions for the 4 selected nodes for either variant.
    """
    apply_style()
    sub = cov_data[disp_key][variant_key]
    coords = res["coords"]
    step = res["step"]
    
    is_true = (disp_key == "true")
    mode_label = "True Displacement (No Noise)" if is_true else "Observed Displacement (with DIC Noise)"
    has_var = (variant_key == "with_var")
    var_label = "WITH Learned Free Residual Variance" if has_var else "WITHOUT Free Residual Variance"
    
    hist_color = "#2ca02c" if has_var else "#1f77b4"
    mean_color = "darkgreen" if has_var else "navy"
    
    fig, axes = plt.subplots(2, 4, figsize=(22.0, 9.5))
    
    for col_idx, node_idx in enumerate(selected_nodes):
        node_coords = coords[node_idx]
        is_fx = node_idx in res["free_x_idx"]
        is_fy = node_idx in res["free_y_idx"]
        
        q025 = sub["q025"][node_idx]
        q975 = sub["q975"][node_idx]
        
        if has_var:
            r_node_x = sub["R_with_x"][:, node_idx]
            r_node_y = sub["R_with_y"][:, node_idx]
        else:
            r_node_x = sub["R_all"][:, node_idx, 0]
            r_node_y = sub["R_all"][:, node_idx, 1]
            
        means = [float(np.mean(r_node_x)), float(np.mean(r_node_y))]
        stds = [float(np.std(r_node_x)), float(np.std(r_node_y))]
        
        cov_x = (q025[0] <= 0.0 <= q975[0]) if is_fx else True
        cov_y = (q025[1] <= 0.0 <= q975[1]) if is_fy else True
        is_cov = cov_x and cov_y
        
        node_status_str = "COVERED" if is_cov else "UNCOVERED"
        badge_color = "#28a745" if is_cov else "#dc3545"
        
        for row_idx, (dim_label, r_data, mu, sigma, q_lo, q_hi, is_free, is_c) in enumerate([
            (r"R_x", r_node_x, means[0], stds[0], q025[0], q975[0], is_fx, cov_x),
            (r"R_y", r_node_y, means[1], stds[1], q025[1], q975[1], is_fy, cov_y),
        ]):
            ax = axes[row_idx, col_idx]
            
            # Histogram
            ax.hist(r_data, bins=35, density=True, alpha=0.35, color=hist_color, edgecolor="black", lw=0.5)
            
            # Smooth KDE
            try:
                sub_r = np.random.choice(r_data, size=min(len(r_data), 4000), replace=False)
                kde = gaussian_kde(sub_r)
                span = np.percentile(r_data, 99.5) - np.percentile(r_data, 0.5)
                pad = max(span * 0.15, 1e-6)
                gx = np.linspace(np.percentile(r_data, 0.5) - pad, np.percentile(r_data, 99.5) + pad, 300)
                ax.plot(gx, kde(gx), color=hist_color, lw=2.2, label="Predictive Posterior")
            except Exception:
                pass
            
            # Target line R = 0
            ax.axvline(0.0, color="red", linestyle="--", lw=1.8, label="Equilibrium ($R=0$)")
            ax.axvline(mu, color=mean_color, linestyle="-", lw=1.5, label=rf"Mean: ${format_sci(mu)}$")
            ax.axvspan(q_lo, q_hi, color=hist_color, alpha=0.15, label=r"95% CI $[q_{0.025}, q_{0.975}]$")
            
            ax.set_xlabel(rf"Residual ${dim_label}$ [N]", fontsize=12)
            ax.set_ylabel("Probability Density", fontsize=12)
            
            status_text = "Covered" if is_c else "Uncovered"
            
            if row_idx == 0:
                ax.set_title(
                    rf"$\mathbf{{Node\ {node_idx}}}$ ({node_status_str})" + "\n" +
                    rf"$(X={node_coords[0]:.3f},\ Y={node_coords[1]:.3f})\ \mathrm{{m}}$",
                    fontsize=13, fontweight="bold", pad=10,
                    color=badge_color
                )
            
            box_text = (
                rf"$\mu = {format_sci(mu)}$" + "\n" +
                rf"$\sigma = {format_sci(sigma)}$" + "\n" +
                rf"$95\%\ \mathrm{{CI}}: [{format_sci(q_lo)}, {format_sci(q_hi)}]$" + "\n" +
                rf"$\mathbf{{Status:}}$ {status_text}"
            )
            ax.text(
                0.04, 0.95, box_text, transform=ax.transAxes, verticalalignment="top",
                fontsize=8.5, bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.92, edgecolor="#cccccc")
            )
            ax.grid(True, linestyle="--", alpha=0.3)
            
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper right", fontsize=8.0, framealpha=0.9)
    
    fig.suptitle(
        rf"Local Free Node Equilibrium Residual Distributions at Step {step} (BLOCK)" + "\n" +
        rf"{mode_label} — [{var_label}]",
        fontsize=16, fontweight="bold", y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_pdf = os.path.join(save_path, f"local_node_residuals_step{step}_{disp_key}_{variant_key}.pdf")
    save_figure(fig, out_pdf, make_png=True)
    plt.close(fig)
    print(f"✅ Saved local node plot: {out_pdf}")


def plot_local_nodes_variance_overlay(res, cov_data, selected_nodes, disp_key, save_path):
    """
    Overlays the residual distributions WITHOUT variance (Blue) and WITH variance (Green)
    in each subplot for the 4 selected nodes.
    """
    apply_style()
    sub_no = cov_data[disp_key]["without_var"]
    sub_with = cov_data[disp_key]["with_var"]
    coords = res["coords"]
    step = res["step"]
    
    c_no = "#1f77b4"   # Steel Blue (Without variance)
    c_with = "#2ca02c" # Forest Green (With variance)
    
    fig, axes = plt.subplots(2, 4, figsize=(23.0, 10.0))
    
    for col_idx, node_idx in enumerate(selected_nodes):
        node_coords = coords[node_idx]
        is_fx = node_idx in res["free_x_idx"]
        is_fy = node_idx in res["free_y_idx"]
        
        # Without stats
        rx_no = sub_no["R_all"][:, node_idx, 0]
        ry_no = sub_no["R_all"][:, node_idx, 1]
        q025_no = sub_no["q025"][node_idx]
        q975_no = sub_no["q975"][node_idx]
        cov_x_no = (q025_no[0] <= 0.0 <= q975_no[0]) if is_fx else True
        cov_y_no = (q025_no[1] <= 0.0 <= q975_no[1]) if is_fy else True
        is_cov_no = cov_x_no and cov_y_no
        
        # With stats
        rx_with = sub_with["R_with_x"][:, node_idx]
        ry_with = sub_with["R_with_y"][:, node_idx]
        q025_with = sub_with["q025"][node_idx]
        q975_with = sub_with["q975"][node_idx]
        cov_x_with = (q025_with[0] <= 0.0 <= q975_with[0]) if is_fx else True
        cov_y_with = (q025_with[1] <= 0.0 <= q975_with[1]) if is_fy else True
        is_cov_with = cov_x_with and cov_y_with
        
        status_change = f"No: {'Cov' if is_cov_no else 'Uncov'} $\\to$ With: {'Cov' if is_cov_with else 'Uncov'}"
        header_color = "#28a745" if is_cov_with else "#dc3545"
        
        for row_idx, (dim_label, r_n, r_w, qln, qhn, qlw, qhw, cn, cw) in enumerate([
            (r"R_x", rx_no, rx_with, q025_no[0], q975_no[0], q025_with[0], q975_with[0], cov_x_no, cov_x_with),
            (r"R_y", ry_no, ry_with, q025_no[1], q975_no[1], q025_with[1], q975_with[1], cov_y_no, cov_y_with),
        ]):
            ax = axes[row_idx, col_idx]
            
            # Histograms
            ax.hist(r_n, bins=35, density=True, alpha=0.28, color=c_no, edgecolor="navy", lw=0.4)
            ax.hist(r_w, bins=35, density=True, alpha=0.25, color=c_with, edgecolor="darkgreen", lw=0.4)
            
            # KDEs
            try:
                kde_n = gaussian_kde(r_n)
                sub_rw = np.random.choice(r_w, size=min(len(r_w), 4000), replace=False)
                kde_w = gaussian_kde(sub_rw)
                all_vals = np.concatenate([r_n, sub_rw, [0.0]])
                gx = np.linspace(np.percentile(all_vals, 0.2), np.percentile(all_vals, 99.8), 350)
                ax.plot(gx, kde_n(gx), color=c_no, lw=2.2, label=r"Without $\sigma_{\mathrm{free}}^2$")
                ax.plot(gx, kde_w(gx), color=c_with, lw=2.2, linestyle="--", label=r"With $\sigma_{\mathrm{free}}^2$")
            except Exception:
                pass
            
            # Target line R = 0
            ax.axvline(0.0, color="red", linestyle=":", lw=2.0, label="Equilibrium ($R=0$)")
            
            # CIs
            ax.axvspan(qln, qhn, color=c_no, alpha=0.12, label="95% CI (No Var)")
            ax.axvspan(qlw, qhw, color=c_with, alpha=0.12, label=r"95% CI (With $\sigma_{\mathrm{free}}^2$)")
            
            ax.set_xlabel(rf"Residual ${dim_label}$ [N]", fontsize=12)
            ax.set_ylabel("Probability Density", fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.3)
            
            if row_idx == 0:
                ax.set_title(
                    rf"$\mathbf{{Node\ {node_idx}}}$ ({status_change})" + "\n" +
                    rf"$(X={node_coords[0]:.3f},\ Y={node_coords[1]:.3f})\ \mathrm{{m}}$",
                    fontsize=12.5, fontweight="bold", pad=10, color=header_color
                )
            
            box_text = (
                rf"$\mathbf{{Without\ Var:}}\ {'Cov' if cn else 'Uncov'}\ (\sigma={format_sci(np.std(r_n))})$" + "\n"
                rf"$95\%\ \mathrm{{CI}}: [{format_sci(qln)}, {format_sci(qhn)}]$" + "\n"
                rf"$\mathbf{{With\ Var:}}\ {'Cov' if cw else 'Uncov'}\ (\sigma={format_sci(np.std(r_w))})$" + "\n"
                rf"$95\%\ \mathrm{{CI}}: [{format_sci(qlw)}, {format_sci(qhw)}]$"
            )
            ax.text(
                0.04, 0.95, box_text, transform=ax.transAxes, verticalalignment="top",
                fontsize=7.8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.92, edgecolor="#cccccc")
            )
            
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper right", fontsize=7.5, framealpha=0.92)
                
    disp_title = "Observed Displacement (with DIC Noise)" if disp_key == "exp" else "True Displacement (No Noise)"
    fig.suptitle(
        rf"Local Residual Distribution Comparison: WITHOUT vs WITH Learned Free Residual Variance" + "\n" +
        rf"{disp_title} at Load Step {step}",
        fontsize=15, fontweight="bold", y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    out_pdf = os.path.join(save_path, f"local_node_residuals_step{step}_{disp_key}_variance_comparison.pdf")
    save_figure(fig, out_pdf, make_png=True)
    plt.close(fig)
    print(f"✅ Saved local variance overlay plot: {out_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Incorporate free residual variance into equilibrium coverage")
    parser.add_argument(
        "--data_file",
        type=str,
        default="results/20261001T223130_isihara_0.0001_0.02_1.0_0.5_5_1.0_isotropic_block/1/fem_validation/block/fem_distilled_samples.npz"
    )
    parser.add_argument("--step", type=int, default=19)
    parser.add_argument(
        "--save_path",
        type=str,
        default="results/20261001T223130_isihara_0.0001_0.02_1.0_0.5_5_1.0_isotropic_block/1/fem_validation/block"
    )
    parser.add_argument(
        "--extraction_metrics",
        type=str,
        default="results/20261001T223130_isihara_0.0001_0.02_1.0_0.5_5_1.0_isotropic_block/1/extracted/extraction_metrics.json"
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_mc_draws", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    # 1. Load learned free residual standard deviations
    with open(args.extraction_metrics, "r") as f:
        em = json.load(f)
    sigma_free_x = float(em["sigma_free_x"])
    sigma_free_y = float(em["sigma_free_y"])
    print(f"📈 Loaded learned free residual noise parameters:")
    print(f"   sigma_free_x: {sigma_free_x:.6f}  (variance: {sigma_free_x**2:.6e})")
    print(f"   sigma_free_y: {sigma_free_y:.6f}  (variance: {sigma_free_y**2:.6e})")

    # 2. Compute base residuals across 1024 samples
    print("⏳ Computing deterministic nodal residuals across 1024 parameter samples...")
    res = compute_residuals_and_coverage(args.data_file, step=args.step, batch_size=args.batch_size)

    # 3. Evaluate coverage with and without learned residual variance
    print("⏳ Evaluating coverage with and without learned residual variance...")
    cov_data = evaluate_coverage_with_and_without_variance(
        res, sigma_free_x, sigma_free_y, n_mc_draws=args.n_mc_draws
    )

    # Print summary metrics table
    print("\n" + "="*70)
    print(f"EQUILIBRIUM COVERAGE METRICS AT LOAD STEP {args.step} (BLOCK)")
    print("="*70)
    for disp in ["exp", "true"]:
        sub_no = cov_data[disp]["without_var"]
        sub_with = cov_data[disp]["with_var"]
        print(f"\n--- {disp.upper()} DISPLACEMENT ---")
        print(f"  WITHOUT sigma_free^2:")
        print(f"    Nodal Coverage: {sub_no['cov_nodal_pct']:.2f}% ({sub_no['n_cov_nodes']}/{len(res['free_nodes_idx'])})")
        print(f"    DoF Coverage:   {sub_no['cov_dof_pct']:.2f}% (X: {sub_no['cov_x_pct']:.2f}%, Y: {sub_no['cov_y_pct']:.2f}%)")
        print(f"  WITH sigma_free^2:")
        print(f"    Nodal Coverage: {sub_with['cov_nodal_pct']:.2f}% ({sub_with['n_cov_nodes']}/{len(res['free_nodes_idx'])})")
        print(f"    DoF Coverage:   {sub_with['cov_dof_pct']:.2f}% (X: {sub_with['cov_x_pct']:.2f}%, Y: {sub_with['cov_y_pct']:.2f}%)")
    print("="*70 + "\n")

    # Save metrics JSON
    metrics_export = {
        "step": args.step,
        "sigma_free_x": sigma_free_x,
        "sigma_free_y": sigma_free_y,
        "variance_free_x": sigma_free_x**2,
        "variance_free_y": sigma_free_y**2,
        "exp": {
            "without_variance": {
                "n_cov_nodes": cov_data["exp"]["without_var"]["n_cov_nodes"],
                "n_uncov_nodes": cov_data["exp"]["without_var"]["n_uncov_nodes"],
                "cov_nodal_pct": cov_data["exp"]["without_var"]["cov_nodal_pct"],
                "cov_dof_pct": cov_data["exp"]["without_var"]["cov_dof_pct"],
                "cov_x_pct": cov_data["exp"]["without_var"]["cov_x_pct"],
                "cov_y_pct": cov_data["exp"]["without_var"]["cov_y_pct"],
            },
            "with_variance": {
                "n_cov_nodes": cov_data["exp"]["with_var"]["n_cov_nodes"],
                "n_uncov_nodes": cov_data["exp"]["with_var"]["n_uncov_nodes"],
                "cov_nodal_pct": cov_data["exp"]["with_var"]["cov_nodal_pct"],
                "cov_dof_pct": cov_data["exp"]["with_var"]["cov_dof_pct"],
                "cov_x_pct": cov_data["exp"]["with_var"]["cov_x_pct"],
                "cov_y_pct": cov_data["exp"]["with_var"]["cov_y_pct"],
            }
        },
        "true": {
            "without_variance": {
                "n_cov_nodes": cov_data["true"]["without_var"]["n_cov_nodes"],
                "n_uncov_nodes": cov_data["true"]["without_var"]["n_uncov_nodes"],
                "cov_nodal_pct": cov_data["true"]["without_var"]["cov_nodal_pct"],
                "cov_dof_pct": cov_data["true"]["without_var"]["cov_dof_pct"],
                "cov_x_pct": cov_data["true"]["without_var"]["cov_x_pct"],
                "cov_y_pct": cov_data["true"]["without_var"]["cov_y_pct"],
            },
            "with_variance": {
                "n_cov_nodes": cov_data["true"]["with_var"]["n_cov_nodes"],
                "n_uncov_nodes": cov_data["true"]["with_var"]["n_uncov_nodes"],
                "cov_nodal_pct": cov_data["true"]["with_var"]["cov_nodal_pct"],
                "cov_dof_pct": cov_data["true"]["with_var"]["cov_dof_pct"],
                "cov_x_pct": cov_data["true"]["with_var"]["cov_x_pct"],
                "cov_y_pct": cov_data["true"]["with_var"]["cov_y_pct"],
            }
        }
    }
    with open(os.path.join(args.save_path, f"coverage_metrics_step{args.step}_variance_comparison.json"), "w") as f:
        json.dump(metrics_export, f, indent=4)

    # 4. Generate 4-panel plots (without, with, comparison) for EXP
    print("📊 Generating 4-panel plots for Observed displacement...")
    plot_four_panel_variant(res, cov_data, "exp", "without_var", sigma_free_x, sigma_free_y, args.save_path)
    plot_four_panel_variant(res, cov_data, "exp", "with_var", sigma_free_x, sigma_free_y, args.save_path)
    plot_side_by_side_residuals_coverage_comparison(res, cov_data, "exp", args.save_path)

    # 5. Generate 4-panel plots for TRUE
    print("📊 Generating 4-panel plots for True displacement...")
    plot_four_panel_variant(res, cov_data, "true", "without_var", sigma_free_x, sigma_free_y, args.save_path)
    plot_four_panel_variant(res, cov_data, "true", "with_var", sigma_free_x, sigma_free_y, args.save_path)
    plot_side_by_side_residuals_coverage_comparison(res, cov_data, "true", args.save_path)

    # 6. Generate local node distribution plots for the 4 selected nodes
    selected_nodes = [9, 10, 91, 114]
    print(f"📊 Generating local node distribution plots for nodes {selected_nodes}...")
    plot_local_nodes_variant(res, cov_data, selected_nodes, "exp", "without_var", sigma_free_x, sigma_free_y, args.save_path)
    plot_local_nodes_variant(res, cov_data, selected_nodes, "exp", "with_var", sigma_free_x, sigma_free_y, args.save_path)
    plot_local_nodes_variance_overlay(res, cov_data, selected_nodes, "exp", args.save_path)

    # For true as well
    plot_local_nodes_variant(res, cov_data, selected_nodes, "true", "without_var", sigma_free_x, sigma_free_y, args.save_path)
    plot_local_nodes_variant(res, cov_data, selected_nodes, "true", "with_var", sigma_free_x, sigma_free_y, args.save_path)
    plot_local_nodes_variance_overlay(res, cov_data, selected_nodes, "true", args.save_path)

    print("🎉 All variance-aware coverage plots and analyses completed successfully!")


if __name__ == "__main__":
    main()
