#!/usr/bin/env python3
"""
plots/plot_selected_node_residuals.py

Plots the posterior distribution of nodal equilibrium residuals R = f_int across
distilled parameter samples for 4 selected nodes:
- 2 Covered Nodes: Node 9 and Node 10 (covered in both True and Observed displacement fields)
- 2 Uncovered Nodes: Node 91 (notch concentration) and Node 114 (bulk plate interior)

Generates:
1. True Displacement Residual Distributions (2x4 grid: Rx and Ry for 4 nodes)
2. Observed Displacement Residual Distributions (2x4 grid)
3. Direct Comparison Overlay (True vs Observed for all 4 nodes)
4. Domain Location Map highlighting the 4 selected nodes on the block mesh
"""

import os
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


def plot_node_residuals_single_mode(res, selected_nodes, disp_key, save_path):
    """
    Plots a 2x4 grid of residual distributions (Rx and Ry) for the 4 selected nodes
    under either 'true' or 'exp' displacement.
    """
    apply_style()
    sub = res[disp_key]
    coords = res["coords"]
    step = res["step"]
    
    is_true = (disp_key == "true")
    mode_label = "True Displacement (No Noise)" if is_true else "Observed Displacement (with DIC Noise)"
    file_suffix = "true" if is_true else "exp"
    hist_color = "#1f77b4" if is_true else "#2ca02c"
    mean_color = "navy" if is_true else "darkgreen"
    
    fig, axes = plt.subplots(2, 4, figsize=(22.0, 9.5))
    
    for col_idx, node_idx in enumerate(selected_nodes):
        node_coords = coords[node_idx]
        is_fx = node_idx in res["free_x_idx"]
        is_fy = node_idx in res["free_y_idx"]
        
        R_node = sub["R_all"][:, node_idx, :] # (1024, 2)
        q025 = sub["q025"][node_idx]
        q975 = sub["q975"][node_idx]
        means = np.mean(R_node, axis=0)
        stds = np.std(R_node, axis=0)
        
        cov_x = (q025[0] <= 0.0 <= q975[0]) if is_fx else True
        cov_y = (q025[1] <= 0.0 <= q975[1]) if is_fy else True
        is_cov = cov_x and cov_y
        
        node_status_str = "COVERED" if is_cov else "UNCOVERED"
        badge_color = "#28a745" if is_cov else "#dc3545"
        
        for row_idx, (dim_label, r_data, mu, sigma, q_lo, q_hi, is_free, is_c) in enumerate([
            (r"R_x", R_node[:, 0], means[0], stds[0], q025[0], q975[0], is_fx, cov_x),
            (r"R_y", R_node[:, 1], means[1], stds[1], q025[1], q975[1], is_fy, cov_y),
        ]):
            ax = axes[row_idx, col_idx]
            
            # Histogram
            ax.hist(r_data, bins=35, density=True, alpha=0.35, color=hist_color, edgecolor="black", lw=0.5)
            
            # Smooth KDE
            try:
                kde = gaussian_kde(r_data)
                span = np.percentile(r_data, 99.5) - np.percentile(r_data, 0.5)
                pad = max(span * 0.15, 1e-6)
                gx = np.linspace(np.percentile(r_data, 0.5) - pad, np.percentile(r_data, 99.5) + pad, 300)
                ax.plot(gx, kde(gx), color=hist_color, lw=2.2, label="Distilled Posterior")
            except Exception:
                pass
            
            # Equilibrium line R = 0
            ax.axvline(0.0, color="red", linestyle="--", lw=1.8, label="Equilibrium ($R=0$)")
            
            # Mean line
            ax.axvline(mu, color=mean_color, linestyle="-", lw=1.5, label=rf"Mean: ${format_sci(mu)}$")
            
            # 95% Credible Interval span
            ax.axvspan(q_lo, q_hi, color=hist_color, alpha=0.15, label=r"95% CI $[q_{0.025}, q_{0.975}]$")
            
            # Axes labels and limits
            ax.set_xlabel(rf"Residual ${dim_label}$ [N]", fontsize=12)
            ax.set_ylabel("Probability Density", fontsize=12)
            
            status_text = "Covered" if is_c else "Uncovered"
            status_color = "darkgreen" if is_c else "red"
            
            # Title only on top row
            if row_idx == 0:
                ax.set_title(
                    rf"$\mathbf{{Node\ {node_idx}}}$ ({node_status_str})" + "\n" +
                    rf"$(X={node_coords[0]:.3f},\ Y={node_coords[1]:.3f})\ \mathrm{{m}}$",
                    fontsize=13, fontweight="bold", pad=10,
                    color=badge_color
                )
            
            # Metrics text box
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
            
            # Include legend only on first subplot to keep clean
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper right", fontsize=8.0, framealpha=0.9)
    
    fig.suptitle(
        rf"Local Free Node Equilibrium Residual Distributions at Step {step} (BLOCK)" + "\n" +
        rf"{mode_label}",
        fontsize=16, fontweight="bold", y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_pdf = os.path.join(save_path, f"local_node_residuals_step{step}_{file_suffix}.pdf")
    save_figure(fig, out_pdf, make_png=True)
    plt.close(fig)
    print(f"✅ Saved single mode plot: {out_pdf}")


def plot_node_residuals_comparison(res, selected_nodes, save_path):
    """
    Plots a 2x4 grid comparing True Displacement vs Observed Displacement (Exp)
    residual distributions overlaid on the same axes for all 4 nodes.
    """
    apply_style()
    coords = res["coords"]
    step = res["step"]
    
    c_true = "#1f77b4" # Blue
    c_exp = "#d95f02"  # Vermillion / Orange
    
    fig, axes = plt.subplots(2, 4, figsize=(23.0, 10.0))
    
    for col_idx, node_idx in enumerate(selected_nodes):
        node_coords = coords[node_idx]
        is_fx = node_idx in res["free_x_idx"]
        is_fy = node_idx in res["free_y_idx"]
        
        # True stats
        sub_t = res["true"]
        Rt = sub_t["R_all"][:, node_idx, :]
        q025_t = sub_t["q025"][node_idx]
        q975_t = sub_t["q975"][node_idx]
        mu_t = np.mean(Rt, axis=0)
        std_t = np.std(Rt, axis=0)
        cov_xt = (q025_t[0] <= 0.0 <= q975_t[0]) if is_fx else True
        cov_yt = (q025_t[1] <= 0.0 <= q975_t[1]) if is_fy else True
        is_cov_t = cov_xt and cov_yt
        
        # Exp stats
        sub_e = res["exp"]
        Re = sub_e["R_all"][:, node_idx, :]
        q025_e = sub_e["q025"][node_idx]
        q975_e = sub_e["q975"][node_idx]
        mu_e = np.mean(Re, axis=0)
        std_e = np.std(Re, axis=0)
        cov_xe = (q025_e[0] <= 0.0 <= q975_e[0]) if is_fx else True
        cov_ye = (q025_e[1] <= 0.0 <= q975_e[1]) if is_fy else True
        is_cov_e = cov_xe and cov_ye
        
        node_cat = "Covered Node" if (is_cov_t and is_cov_e) else "Uncovered Node"
        header_color = "#28a745" if (is_cov_t and is_cov_e) else "#dc3545"
        
        for row_idx, (dim_label, r_t, r_e, mt, me, st, se, qlt, qht, qle, qhe, ct, ce) in enumerate([
            (r"R_x", Rt[:, 0], Re[:, 0], mu_t[0], mu_e[0], std_t[0], std_e[0], q025_t[0], q975_t[0], q025_e[0], q975_e[0], cov_xt, cov_xe),
            (r"R_y", Rt[:, 1], Re[:, 1], mu_t[1], mu_e[1], std_t[1], std_e[1], q025_t[1], q975_t[1], q025_e[1], q975_e[1], cov_yt, cov_ye),
        ]):
            ax = axes[row_idx, col_idx]
            
            # Histograms
            ax.hist(r_t, bins=35, density=True, alpha=0.28, color=c_true, edgecolor="navy", lw=0.4)
            ax.hist(r_e, bins=35, density=True, alpha=0.28, color=c_exp, edgecolor="darkred", lw=0.4)
            
            # KDEs
            try:
                kde_t = gaussian_kde(r_t)
                kde_e = gaussian_kde(r_e)
                all_vals = np.concatenate([r_t, r_e, [0.0]])
                gx = np.linspace(np.percentile(all_vals, 0.2), np.percentile(all_vals, 99.8), 350)
                ax.plot(gx, kde_t(gx), color=c_true, lw=2.2, label="True Disp. (No Noise)")
                ax.plot(gx, kde_e(gx), color=c_exp, lw=2.2, linestyle="--", label="Observed Disp. (DIC Noise)")
            except Exception:
                pass
            
            # Target line R = 0
            ax.axvline(0.0, color="red", linestyle=":", lw=2.0, label="Equilibrium ($R=0$)")
            
            # CIs
            ax.axvspan(qlt, qht, color=c_true, alpha=0.12, label="True 95% CI")
            ax.axvspan(qle, qhe, color=c_exp, alpha=0.12, label="Observed 95% CI")
            
            ax.set_xlabel(rf"Residual ${dim_label}$ [N]", fontsize=12)
            ax.set_ylabel("Probability Density", fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.3)
            
            if row_idx == 0:
                ax.set_title(
                    rf"$\mathbf{{Node\ {node_idx}}}$ ({node_cat})" + "\n" +
                    rf"$(X={node_coords[0]:.3f},\ Y={node_coords[1]:.3f})\ \mathrm{{m}}$",
                    fontsize=13, fontweight="bold", pad=10, color=header_color
                )
            
            # Inset metrics comparison
            t_status = "Cov" if ct else "Uncov"
            e_status = "Cov" if ce else "Uncov"
            box_text = (
                rf"$\mathbf{{True:}}\ {t_status} \mid \mu={format_sci(mt)}$" + "\n" +
                rf"$95\%\ \mathrm{{CI}}: [{format_sci(qlt)}, {format_sci(qht)}]$" + "\n" +
                rf"$\mathbf{{Obs.:}}\ {e_status} \mid \mu={format_sci(me)}$" + "\n" +
                rf"$95\%\ \mathrm{{CI}}: [{format_sci(qle)}, {format_sci(qhe)}]$"
            )
            ax.text(
                0.04, 0.95, box_text, transform=ax.transAxes, verticalalignment="top",
                fontsize=8.0, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.92, edgecolor="#cccccc")
            )
            
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper right", fontsize=7.5, framealpha=0.92)
                
    fig.suptitle(
        rf"Local Free Node Equilibrium Residual Comparison: True vs Observed (Step {step})" + "\n" +
        r"$\mathbf{Covered\ Nodes}$ (9, 10: Notch) vs $\mathbf{Uncovered\ Nodes}$ (91: Near-Notch, 114: Plate Interior)",
        fontsize=16, fontweight="bold", y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    out_pdf = os.path.join(save_path, f"local_node_residuals_step{step}_comparison.pdf")
    save_figure(fig, out_pdf, make_png=True)
    plt.close(fig)
    print(f"✅ Saved comparison plot: {out_pdf}")


def plot_node_spatial_locations(res, selected_nodes, save_path):
    """
    Plots the domain mesh highlighting the 4 selected nodes with distinct callouts and annotations.
    """
    apply_style()
    coords = res["coords"]
    cells = res["cells"]
    step = res["step"]
    triang = tri.Triangulation(coords[:, 0], coords[:, 1], cells)
    
    fig, ax = plt.subplots(figsize=(10.5, 9.0))
    
    # Mesh
    ax.triplot(triang, color="#d8d8d8", lw=0.6, zorder=1)
    
    # All free nodes faintly in background
    ax.scatter(
        coords[res["free_nodes_idx"], 0], coords[res["free_nodes_idx"], 1],
        c="#e0e0e0", marker="o", s=30, edgecolors="#b0b0b0", lw=0.5,
        label="Other Free Nodes", zorder=2
    )
    
    # Dirichlet nodes
    dir_idx = res["dirichlet_nodes_idx"]
    if len(dir_idx) > 0:
        ax.scatter(
            coords[dir_idx, 0], coords[dir_idx, 1],
            c="#6c757d", marker="s", s=55, edgecolors="black", lw=0.8,
            label=f"Constrained Dirichlet ({len(dir_idx)})", zorder=3
        )
    
    # Selected covered nodes
    cov_sel = [n for n in selected_nodes if n in [9, 10]]
    ax.scatter(
        coords[cov_sel, 0], coords[cov_sel, 1],
        c="#28a745", marker="o", s=140, edgecolors="#155724", lw=1.8,
        label=f"Selected Covered Nodes (9, 10)", zorder=5
    )
    
    # Selected uncovered nodes
    uncov_sel = [n for n in selected_nodes if n in [91, 114]]
    ax.scatter(
        coords[uncov_sel, 0], coords[uncov_sel, 1],
        c="#dc3545", marker="o", s=140, edgecolors="#721c24", lw=1.8,
        label=f"Selected Uncovered Nodes (91, 114)", zorder=5
    )
    
    # Callout annotations for the 4 nodes
    callouts = {
        9: (r"$\mathbf{Node\ 9\ [Covered]}$" + "\n" + r"$(0.056, 0.083)\ \mathrm{m}$" + "\n" + r"Notch boundary", (0.16, 0.17)),
        10: (r"$\mathbf{Node\ 10\ [Covered]}$" + "\n" + r"$(0.038, 0.092)\ \mathrm{m}$" + "\n" + r"Notch boundary", (-0.02, 0.22)),
        91: (r"$\mathbf{Node\ 91\ [Uncovered]}$" + "\n" + r"$(0.092, 0.100)\ \mathrm{m}$" + "\n" + r"Shear / Notch concentration", (0.22, 0.07)),
        114: (r"$\mathbf{Node\ 114\ [Uncovered]}$" + "\n" + r"$(0.497, 0.518)\ \mathrm{m}$" + "\n" + r"Plate center / Bulk interior", (0.55, 0.58)),
    }
    
    for n_id, (txt, text_pos) in callouts.items():
        n_xy = coords[n_id]
        box_c = "#d4edda" if n_id in cov_sel else "#f8d7da"
        edge_c = "#28a745" if n_id in cov_sel else "#dc3545"
        ax.annotate(
            txt,
            xy=(n_xy[0], n_xy[1]),
            xytext=text_pos,
            arrowprops=dict(arrowstyle="->", color=edge_c, lw=1.5, shrinkA=3, shrinkB=4),
            bbox=dict(boxstyle="round,pad=0.4", facecolor=box_c, edgecolor=edge_c, alpha=0.95),
            fontsize=9.5, fontweight="bold", zorder=6
        )
    
    ax.set_aspect("equal")
    ax.set_xlim(-0.08, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"Material Coordinate $X$ [m]", fontsize=12)
    ax.set_ylabel(r"Material Coordinate $Y$ [m]", fontsize=12)
    ax.set_title(
        rf"Domain Locations of Selected Analyzed Nodes (Step {step})" + "\n" +
        r"2 Covered Nodes (9, 10) and 2 Uncovered Nodes (91, 114)",
        fontsize=14, fontweight="bold", pad=12
    )
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.92, facecolor="white")
    
    plt.tight_layout()
    out_pdf = os.path.join(save_path, f"local_node_residuals_step{step}_locations.pdf")
    save_figure(fig, out_pdf, make_png=True)
    plt.close(fig)
    print(f"✅ Saved spatial locations map: {out_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Plot local node residual distributions")
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
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    
    print("⏳ Computing equilibrium residuals and coverage...")
    res = compute_residuals_and_coverage(args.data_file, step=args.step, batch_size=args.batch_size)
    
    # Selected nodes: 2 covered (9, 10), 2 uncovered (91, 114)
    selected_nodes = [9, 10, 91, 114]
    
    print("📊 Generating True Displacement Residual Distributions Plot...")
    plot_node_residuals_single_mode(res, selected_nodes, "true", args.save_path)
    
    print("📊 Generating Observed Displacement Residual Distributions Plot...")
    plot_node_residuals_single_mode(res, selected_nodes, "exp", args.save_path)
    
    print("📊 Generating Comparison Residual Distributions Plot...")
    plot_node_residuals_comparison(res, selected_nodes, args.save_path)
    
    print("🗺️ Generating Spatial Node Locations Map...")
    plot_node_spatial_locations(res, selected_nodes, args.save_path)
    
    print("🎉 All plots generated successfully!")


if __name__ == "__main__":
    main()
