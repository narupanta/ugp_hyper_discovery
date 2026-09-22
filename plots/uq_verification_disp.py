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

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import r2_score
from scipy.stats import norm

from plots.theme import apply_style, save_figure

def plot_comprehensive_analysis(u_true, u_pred_samples, node_type, node_to_plot, save_path):
    """
    Generates and saves two separate 2x2 figures (X and Y directions).
    Includes statistical text annotations and a zero-line for error distributions.
    """
    apply_style()
    # Filter for free nodes
    if node_type.shape[1] >= 5:
        free_nodes = (node_type[:, 1] != 1) & (node_type[:, 2] != 1) & (node_type[:, 3] != 1) & (node_type[:, 4] != 1)
    else:
        free_nodes = (node_type[:, 1] != 1) & (node_type[:, 2] != 1)
    
    u_pred_free = u_pred_samples[:, free_nodes, :]  # [Samples, Nodes, 2]
    u_true_free = u_true[free_nodes, :]             # [Nodes, 2]
    
    num_samples, num_nodes, _ = u_pred_free.shape
    node_indices = np.arange(num_nodes)
    
    directions = [
        {'idx': 0, 'label': 'x', 'color': 'purple'},
        {'idx': 1, 'label': 'y', 'color': 'teal'}
    ]

    os.makedirs(save_path, exist_ok=True)

    for dir_info in directions:
        d = dir_info['idx']
        label = dir_info['label']
        main_color = dir_info['color']
        u_p = u_pred_samples[:, :, d]
        u_t = u_true[:, d]
        
        u_p_dim = u_pred_free[:, :, d]
        u_t_dim = u_true_free[:, d]
        
        u_std_per_node = np.std(u_p_dim, axis=0)
        u_true_repeated = np.tile(u_t_dim, num_samples)
        u_pred_flat = u_p_dim.flatten()
        errors_flat = (u_t_dim[None, :] - u_p_dim).flatten()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        plt.subplots_adjust(wspace=0.25, hspace=0.3)
        (ax1, ax2), (ax3, ax4) = axes

        # --- PLOT 1: UNCERTAINTY (LINE CHART) ---
        ax1.plot(node_indices, u_std_per_node, color=main_color, linewidth=1.5)
        ax1.fill_between(node_indices, 0, u_std_per_node, color=main_color, alpha=0.15, label=r'$\sigma$ per Node')
        ax1.set_title(f'Predictive Uncertainty - {label.upper()}', fontsize=13)
        ax1.set_xlabel('Node Index')
        ax1.set_ylabel('Standard Deviation')
        ax1.grid(alpha=0.3, linestyle='--')

        # --- PLOT 2: GLOBAL ACCURACY ---
        ax2.scatter(u_true_repeated, u_pred_flat, color='seagreen', s=1, alpha=0.05)
        mn, mx = u_t_dim.min(), u_t_dim.max()
        ax2.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect Fit')
        ax2.set_title(f'Sample-wise Accuracy - {label.upper()}', fontsize=13)
        ax2.set_xlabel(f'True $u_{label}$')
        ax2.set_ylabel(f'Predicted $u_{label}$')
        ax2.legend(markerscale=10)

        # --- PLOT 3: LOCAL DISTRIBUTION (Selected Node) ---
        node_samples = u_p[:, node_to_plot]
        node_true = u_t[node_to_plot]
        n_mean = np.mean(node_samples)
        n_std = np.std(node_samples)
        
        ax3.hist(node_samples, bins=40, density=True, alpha=0.3, color='dodgerblue', label='Samples')
        ax3.axvline(node_true, color='red', linestyle='--', label=f'True: {node_true:.4f}')
        
        # Add stats text to Plot 3
        stats_text_local = f'Mean: {n_mean:.4e}\nStd: {n_std:.4e}'
        ax3.text(0.05, 0.95, stats_text_local, transform=ax3.transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        ax3.set_title(f'Local Dist. at Node {node_to_plot} - {label.upper()}', fontsize=13)
        ax3.set_xlabel('Displacement Value')
        ax3.legend()

        # --- PLOT 4: GLOBAL ERROR DISTRIBUTION ---
        e_mean = np.mean(errors_flat)
        e_std = np.std(errors_flat)
        
        ax4.hist(errors_flat, bins=60, density=True, alpha=0.3, color='orange', label='Error')
        ax4.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.6) # Vertical line at 0
        
        # Add stats text to Plot 4
        stats_text_global = f'Mean Error: {e_mean:.4e}\nError Std: {e_std:.4e}'
        ax4.text(0.05, 0.95, stats_text_global, transform=ax4.transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        ax4.set_title(f'Global Error Distribution - {label.upper()}', fontsize=13)
        ax4.set_xlabel('Error Value (True - Pred)')
        ax4.set_ylabel('Density')

        # Save and Show
        save_file = os.path.join(save_path, f"analysis_{label}_{node_to_plot}_direction.pdf")
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.show()
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_node_distributions(u_true, u_pred_samples, u_pred_piola_traction_samples, node_to_plot, save_path):
    """
    Plots local distributions in a 2-row grid.
    Row 0: X-direction for all nodes
    Row 1: Y-direction for all nodes
    """
    os.makedirs(save_path, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "legend.fontsize": 16,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "text.usetex": False
    })

    num_nodes_to_plot = len(node_to_plot)
    
    # 2 rows, num_nodes columns
    # Flipped figsize: width scales with nodes, height is fixed for 2 rows
    fig, axes = plt.subplots(2, num_nodes_to_plot, 
                             figsize=(5 * num_nodes_to_plot, 14 / 1.5))
    
    # Standardize axes to 2D array [row, col] even if 1 node is plotted
    if num_nodes_to_plot == 1:
        axes = axes.reshape(2, 1)

    directions = [
        {'idx': 0, 'label': 'X', 'color1': 'blue', 'color2': 'dodgerblue'},
        {'idx': 1, 'label': 'Y', 'color1': 'teal', 'color2': 'green'}
    ]

    for node_col, node_idx in enumerate(node_to_plot):
        for d_info in directions:
            row_idx = d_info['idx']
            ax = axes[row_idx, node_col]
            label_suffix = f"({d_info['label']})"
            
            # Data extraction
            samples = u_pred_samples[:, node_idx, row_idx]
            u_true_node = u_true[node_idx, row_idx]
            
            p_mean = np.mean(samples)
            p_low, p_high = np.quantile(samples, [0.025, 0.975])

            if u_pred_piola_traction_samples is not None:
                pt_samples = u_pred_piola_traction_samples[:, node_idx, row_idx]
                pt_mean = np.mean(pt_samples)
                pt_low, pt_high = np.quantile(pt_samples, [0.025, 0.975])
            else:
                pt_samples = None

            # Internal Annotation
            ax.text(0.05, 0.92, f'Node: {node_idx}', transform=ax.transAxes,
                    fontsize=14, fontweight='bold', verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='gray'))

            # Plots
            ax.hist(samples, bins=40, density=True, alpha=0.3, 
                    color=d_info['color1'], label=f'Piola {label_suffix}')
            ax.axvline(u_true_node, color='red', linestyle='-', linewidth=2, label=f'True {label_suffix}')
            ax.axvline(p_mean, color=d_info['color1'], linestyle='--', linewidth=1.5, label=f'Piola Mean {label_suffix}')
            ax.axvspan(p_low, p_high, color=d_info['color1'], alpha=0.1, label=f'Piola 95% CI {label_suffix}')

            if pt_samples is not None:
                ax.hist(pt_samples, bins=40, density=True, alpha=0.3, 
                        color=d_info['color2'], label=f'PT {label_suffix}')
                ax.axvline(pt_mean, color=d_info['color2'], linestyle='--', linewidth=1.5, label=f'PT Mean {label_suffix}')
                ax.axvspan(pt_low, pt_high, color=d_info['color2'], alpha=0.1, label=f'PT 95% CI {label_suffix}')

            # Formatting
            if node_col == 0:
                ax.set_ylabel(f'Density ({d_info["label"]})', fontweight='bold')
            if row_idx == 1:
                ax.set_xlabel('Displacement')
            # if row_idx == 0:
            #     ax.set_title(f'Node Column {node_idx}')

            ax.grid(alpha=0.2)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

    # --- Legend ---
    handles_all, labels_all = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels_all:
                handles_all.append(handle)
                labels_all.append(label)

    # Sort labels so X-direction labels come first, then Y-direction
    # This ensures X occupies the top row and Y occupies the bottom row when ncol is set
    x_indices = [i for i, l in enumerate(labels_all) if "(X)" in l or "True" in l and "(X)" in l]
    y_indices = [i for i, l in enumerate(labels_all) if "(Y)" in l or "True" in l and "(Y)" in l]
    
    # Reorder handles and labels
    sorted_handles = [handles_all[i] for i in x_indices] + [handles_all[i] for i in y_indices]
    sorted_labels = [labels_all[i] for i in x_indices] + [labels_all[i] for i in y_indices]

    # Calculate ncol: number of labels in one direction (should be equal for X and Y)
    num_cols = len(x_indices)

    fig.legend(sorted_handles, sorted_labels, 
               loc='upper center', 
               bbox_to_anchor=(0.5, 1.02), # Moved up slightly to accommodate 2 rows
               ncol=num_cols, 
               frameon=True,
               columnspacing=1.0,
               handletextpad=0.5)

    # Increase the top margin in rect (0.88) to make room for 2 rows of legend
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    save_file = os.path.join(save_path, "local_node_distributions.pdf")
    save_file_png = os.path.join(save_path, "local_node_distributions.png")
    plt.savefig(save_file, bbox_inches='tight', transparent=True)
    plt.savefig(save_file_png, bbox_inches='tight', transparent=True, dpi=300)
    print(f"Local node distributions plot saved to: {save_file} and {save_file_png}")
    plt.close(fig)
# def plot_node_distributions(u_true, u_pred_samples, u_pred_piola_traction_samples, node_to_plot, save_path):
#     """
#     Plots local distributions with 95% Quantile CIs.
#     Layout: Rows = 2 (X, Y), Columns = num_nodes_to_plot.
#     """
#     os.makedirs(save_path, exist_ok=True)
#     plt.rcParams.update({
#     "font.family": "serif",
#     "font.serif": ["Times New Roman", "DejaVu Serif"], # Falls back to DejaVu if Times isn't found
#     "font.size": 16,                # Base font size
#     "axes.titlesize": 18,           # Subplot titles
#     "axes.labelsize": 16,           # X and Y labels
#     "legend.fontsize": 8,          # Legend text
#     "xtick.labelsize": 10,          # Axis tick numbers
#     "ytick.labelsize": 10,
#     "figure.dpi": 600,              # High resolution for the screen and save
#     "savefig.dpi": 600,             # Ensures saved file is high quality
#     "text.usetex": False            # Set to True only if you have a full LaTeX install
#     })
#     # ... [Keep your plt.rcParams update here] ...

#     num_nodes_to_plot = len(node_to_plot)
    
#     # SWAP HERE: 2 rows, num_nodes_to_plot columns
#     # We increase width (14 -> 5 * nodes) and decrease height (5 * nodes -> 8)
#     fig, axes = plt.subplots(2, num_nodes_to_plot, figsize=(5 * num_nodes_to_plot, 8), squeeze=False)
    
#     directions = [
#         {'idx': 0, 'label': 'X', 'color1': 'blue', 'color2': 'dodgerblue'},
#         {'idx': 1, 'label': 'Y', 'color1': 'teal', 'color2': 'green'}
#     ]

#     for j, node_idx in enumerate(node_to_plot):
#         for i, d_info in enumerate(directions):
#             d = d_info['idx']
            
#             # Indexing is now [row, col] -> [direction_index, node_index]
#             ax = axes[i, j]
            
#             # 1. Extract Data
#             samples = u_pred_samples[:, node_idx, d]
#             pt_samples = u_pred_piola_traction_samples[:, node_idx, d]
#             u_true_node = u_true[node_idx, d]
            
#             # 2. Statistics
#             p_mean, pt_mean = np.mean(samples), np.mean(pt_samples)
#             p_low, p_high = np.quantile(samples, [0.025, 0.975])
#             pt_low, pt_high = np.quantile(pt_samples, [0.025, 0.975])

#             # 3. Plot Histograms
#             ax.hist(samples, bins=30, density=True, alpha=0.3, color=d_info['color1'], label='Piola Samples')
#             ax.hist(pt_samples, bins=30, density=True, alpha=0.3, color=d_info['color2'], label='PT Samples')

#             # 4. Vertical Lines & Shaded CI
#             ax.axvline(u_true_node, color='red', linestyle='-', linewidth=2, 
#                        label=f'True: {u_true_node:.4e}')
#             ax.axvline(p_mean, color=d_info['color1'], linestyle='--', linewidth=1.5, 
#                        label=f'Piola Mean: {p_mean:.4e}')
#             ax.axvline(pt_mean, color=d_info['color2'], linestyle='--', linewidth=1.5, 
#                        label=f'PT Mean: {pt_mean:.4e}')

#             # 5. Plot Confidence Intervals as Shaded Regions (95% CI)
#             ax.axvspan(p_low, p_high, color=d_info['color1'], alpha=0.1, 
#                        label='Piola 95% CI')
#             ax.axvspan(pt_low, pt_high, color=d_info['color2'], alpha=0.1, 
#                        label='PT 95% CI')

#             # Optional: Add faint boundary lines for the CIs
#             ax.axvline(p_low, color=d_info['color1'], linestyle=':', alpha=0.5, linewidth=1)
#             ax.axvline(p_high, color=d_info['color1'], linestyle=':', alpha=0.5, linewidth=1)
            
#             # Formatting
#             # Titles only on the first row
#             if i == 0:
#                 ax.set_title(f'Node {node_idx}')
            
#             # Y-labels only on the first column
#             if j == 0:
#                 ax.set_ylabel(f'Displacement {d_info["label"]}\nDensity')
                
#             ax.legend(loc='upper right')
#             ax.grid(alpha=0.2)

#     # plt.suptitle('Local Displacement Distributions', fontsize = 22, y=1.05)
#     plt.tight_layout()
    
#     save_file = os.path.join(save_path, "local_node_distributions_quantile.pdf")
#     plt.savefig(save_file, bbox_inches='tight', transparent=True)
#     print(f"Saved to: {save_file}")
# Example: plot_comprehensive_analysis(u_true, u_pred_samples, 50, "plots/")

# Usage:
# plot_global_std_analysis(u_true, u_pred_samples, "results/")

# Usage:
# plot_global_samples_analysis(u_true, u_pred_samples, "results/")

# Example Usage:
# plot_global_index_analysis(u_true, u_pred_samples, "plots/")
# Example Usage:
# plot_global_analysis(u_true, u_pred_samples, node_coords, "plots/")
import os
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import r2_score

def plot_disp_field(node_coords, cells, u_true, u_pred_mean, u_true_val_flat, u_p_mean, u_p_lower_bound, u_p_upper_bound, save_path, mode_str=""):
    apply_style()

    # --- Data Preparation ---
    coords_true = node_coords + u_true
    coords_pred = node_coords + u_pred_mean
    
    def get_mag(u): return np.linalg.norm(u, axis=1)
    mag_true = get_mag(u_true)
    mag_pred = get_mag(u_pred_mean)
    # Nodal RMSE between true and predicted displacement
    error = np.sqrt(np.mean((u_true - u_pred_mean)**2, axis=-1))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), gridspec_kw={"width_ratios": [1, 1, 1.15]})

    # Shared limits for disp magnitude
    vmin_disp = min(mag_true.min(), mag_pred.min())
    vmax_disp = max(mag_true.max(), mag_pred.max())

    # --- 1. Predicted Displacement Magnitude with Observed Domain Underlay ---
    tri_obs = tri.Triangulation(coords_true[:, 0], coords_true[:, 1], cells)
    tri_pred = tri.Triangulation(coords_pred[:, 0], coords_pred[:, 1], cells)

    # 1a. Background shaded fill of observed domain
    axes[0].tripcolor(tri_obs, mag_true, cmap="Blues", alpha=0.30, vmin=vmin_disp, vmax=vmax_disp, zorder=1)
    # 1b. Clear wireframe grid of observed domain
    axes[0].triplot(tri_obs, color="#444444", linestyle=":", linewidth=0.7, alpha=0.75, zorder=2, label=r"Observed ($\mathbf{u}_{\mathrm{obs}}$)")
    
    # 1c. Predicted displacement field contour on top with slight transparency
    im1 = axes[0].tripcolor(tri_pred, mag_pred, cmap="Blues", alpha=0.88, vmin=vmin_disp, vmax=vmax_disp, zorder=3)
    # 1d. Solid contour boundary of predicted domain
    axes[0].triplot(tri_pred, color="#002b4d", linestyle="-", linewidth=0.35, alpha=0.45, zorder=4)

    axes[0].set_aspect("equal")
    axes[0].axis("off")
    div1 = make_axes_locatable(axes[0])
    cax1 = div1.append_axes("right", size="5%", pad=0.08)
    cbar1 = fig.colorbar(im1, cax=cax1, orientation="vertical")
    cbar1.set_label(r"$\|\mathbf{u}_{\mathrm{pred}}\|$", fontsize=10, fontweight="bold")
    cbar1.ax.tick_params(labelsize=8.5)
    cbar1.locator = ticker.MaxNLocator(nbins=4)
    cbar1.update_ticks()

    # --- 2. Nodal RMSE (Oranges) ---
    im2 = axes[1].tripcolor(tri_pred, error, cmap="Oranges")
    axes[1].set_aspect("equal")
    axes[1].axis("off")
    div2 = make_axes_locatable(axes[1])
    cax2 = div2.append_axes("right", size="5%", pad=0.08)
    cbar2 = fig.colorbar(im2, cax=cax2, orientation="vertical")
    cbar2.set_label(r"$\mathrm{RMSE}$", fontsize=10, fontweight="bold")
    cbar2.ax.tick_params(labelsize=8.5)
    cbar2.locator = ticker.MaxNLocator(nbins=4)
    cbar2.update_ticks()

    # --- 3. Parity & Coverage Plot ---
    ux_true, uy_true = u_true_val_flat[:, 0], u_true_val_flat[:, 1]
    ux_med, uy_med = u_p_mean[:, 0], u_p_mean[:, 1]
    ux_lower, ux_upper = u_p_lower_bound[:, 0], u_p_upper_bound[:, 0]
    uy_lower, uy_upper = u_p_lower_bound[:, 1], u_p_upper_bound[:, 1]

    cov_x = np.mean((ux_true >= ux_lower) & (ux_true <= ux_upper)) * 100
    cov_y = np.mean((uy_true >= uy_lower) & (uy_true <= uy_upper)) * 100
    cov_xy = np.mean(np.concatenate([
        (ux_true >= ux_lower) & (ux_true <= ux_upper),
        (uy_true >= uy_lower) & (uy_true <= uy_upper)
    ])) * 100
    
    r2_x = r2_score(ux_true, ux_med)
    r2_y = r2_score(uy_true, uy_med)
    
    rmse_x = np.sqrt(np.mean((ux_med - ux_true)**2))
    rmse_y = np.sqrt(np.mean((uy_med - uy_true)**2))

    ux_err = [np.maximum(0.0, ux_med - ux_lower), np.maximum(0.0, ux_upper - ux_med)]
    uy_err = [np.maximum(0.0, uy_med - uy_lower), np.maximum(0.0, uy_upper - uy_med)]

    axes[2].errorbar(ux_true, ux_med, yerr=ux_err, fmt="x", color="#0072B2", ecolor="#0072B2",
                     alpha=0.35, label=r"$u_x$ " + f"({cov_x:.1f}%)", markersize=4, capsize=0, elinewidth=0.8)
    axes[2].errorbar(uy_true, uy_med, yerr=uy_err, fmt="o", color="#D55E00", ecolor="#D55E00",
                     alpha=0.35, label=r"$u_y$ " + f"({cov_y:.1f}%)", markersize=3, capsize=0, elinewidth=0.8)

    all_vals = np.concatenate([u_true_val_flat.flatten(), u_p_mean.flatten()])
    limits = [all_vals.min(), all_vals.max()]
    axes[2].plot(limits, limits, "k--", linewidth=1.2, label="Isoline", zorder=5)

    axes[2].set_xlabel(r"$u_{\mathrm{obs}}$", fontsize=11, labelpad=5)
    axes[2].set_ylabel(r"$u_{\mathrm{pred}}$", fontsize=11, labelpad=3)
    axes[2].tick_params(axis="both", which="major", labelsize=8.5)
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(loc="lower right", bbox_to_anchor=(0.98, 0.04), frameon=True, facecolor="white", framealpha=0.9, edgecolor="#cccccc", fontsize=8.0)
    axes[2].xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    axes[2].yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    axes[2].set_box_aspect(1)

    # Observed vs Predicted legend centered directly below Panel 1
    import matplotlib.lines as mlines
    obs_line = mlines.Line2D([], [], color='#444444', linestyle=':', linewidth=1.4, label=r'Observed ($\mathbf{u}_{\mathrm{obs}}$)')
    pred_line = mlines.Line2D([], [], color='#002b4d', linestyle='-', linewidth=1.4, label=r'Predicted ($\mathbf{u}_{\mathrm{pred}}$)')
    fig.legend(handles=[obs_line, pred_line], loc='center', bbox_to_anchor=(0.18, 0.035),
               ncol=2, frameon=False, fontsize=8.5, handlelength=1.6, borderpad=0.1)

    # --- Bottom Text Box Banner across figure width (centered under panels 2 & 3) ---
    stats_banner = (
        rf"$95\%\;\mathrm{{EC}}_{{u}} = \mathbf{{{cov_xy:.1f}\%}}$   $\vert$   "
        rf"$r^2_{{u_x}} = \mathbf{{{r2_x:.4f}}},\; r^2_{{u_y}} = \mathbf{{{r2_y:.4f}}}$   $\vert$   "
        rf"$\mathrm{{RMSE}}_{{u_x}} = \mathbf{{{rmse_x:.4f}}},\; \mathrm{{RMSE}}_{{u_y}} = \mathbf{{{rmse_y:.4f}}}$"
    )
    fig.text(0.57, 0.035, stats_banner, ha="center", va="center", fontsize=9.0,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#f5f5f5", edgecolor="#bbbbbb", lw=0.6))

    plt.subplots_adjust(left=0.03, right=0.97, wspace=0.18, bottom=0.20, top=0.96)
    os.makedirs(save_path, exist_ok=True)
    pdf_file = os.path.join(save_path, "displacement_analysis.pdf")
    png_file = os.path.join(save_path, "displacement_analysis.png")
    plt.savefig(pdf_file, bbox_inches="tight")
    plt.savefig(png_file, bbox_inches="tight", dpi=300)
    print(f"Displacement analysis plot saved to: {pdf_file} and {png_file}")
    plt.close(fig)
def plot_disp_r2_coverage(u_true, u_pred_med, u_pred_lower, u_pred_upper, save_path, suffix="_"):
    apply_style()

    fig, ax = plt.subplots(figsize=(10, 7))
    
    # --- Component Extraction ---
    ux_true, uy_true = u_true[:, 0], u_true[:, 1]
    ux_med, uy_med = u_pred_med[:, 0], u_pred_med[:, 1]
    
    ux_lower, ux_upper = u_pred_lower[:, 0], u_pred_upper[:, 0]
    uy_lower, uy_upper = u_pred_lower[:, 1], u_pred_upper[:, 1]
    
    # --- Statistics Calculation (Separate for X and Y) ---
    def get_stats(true, med, low, high):
        inside = (true >= low) & (true <= high)
        cov = np.mean(inside) * 100
        r2 = r2_score(true, med)
        return cov, r2

    cov_x, r2_x = get_stats(ux_true, ux_med, ux_lower, ux_upper)
    cov_y, r2_y = get_stats(uy_true, uy_med, uy_lower, uy_upper)

    # --- Error Bar Formatting ---
    ux_err = [np.maximum(0.0, ux_med - ux_lower), np.maximum(0.0, ux_upper - ux_med)]
    uy_err = [np.maximum(0.0, uy_med - uy_lower), np.maximum(0.0, uy_upper - uy_med)]

    # 1. Plot Displacement X (Blue, 'x' marker)
    ax.errorbar(ux_true, ux_med, yerr=ux_err, fmt='x', color='#0072B2', ecolor='#0072B2', 
                alpha=0.35, label=f'$u_x$ (Cov: {cov_x:.1f}%)', markersize=5, capsize=0, elinewidth=0.8)
    
    # 2. Plot Displacement Y (Orange/Red, 'o' marker)
    ax.errorbar(uy_true, uy_med, yerr=uy_err, fmt='o', color='#D55E00', ecolor='#D55E00', 
                alpha=0.35, label=f'$u_y$ (Cov: {cov_y:.1f}%)', markersize=4, capsize=0, elinewidth=0.8)

    # 3. Identity line (Black dashed)
    all_vals = np.concatenate([u_true.flatten(), u_pred_med.flatten()])
    limits = [all_vals.min(), all_vals.max()]
    ax.plot(limits, limits, 'k--', linewidth=1.5, label='Isoline', zorder=5)

    # --- Annotation Box ---
    stats_text = (f'Estimated Coverage $X$: {cov_x:.1f}%\n'
                  f'Estimated Coverage $Y$: {cov_y:.1f}%\n'
                  f'$R^2_X$: {r2_x:.4f}\n'
                  f'$R^2_Y$: {r2_y:.4f}')
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12, 
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray', lw=0.5))

    # Formatting
    ax.set_xlabel(r'$u_{\mathrm{obs}}$', fontsize=14)
    ax.set_ylabel(r'$u_{\mathrm{pred}}$', fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', frameon=True, fontsize=12)
    
    # Ticks limit
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

    plt.tight_layout()
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, f"disp_r2_coverage_xy_{suffix}.pdf")
        save_file_png = os.path.join(save_path, f"disp_r2_coverage_xy_{suffix}.png")
        plt.savefig(save_file, bbox_inches='tight')
        plt.savefig(save_file_png, bbox_inches='tight', dpi=300)
        print(f"Plot saved to: {save_file} and {save_file_png}")
    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser(description="Isihara Model Dataset and Training Configuration")

    # Dataset & Model Config
    parser.add_argument('--model_path', type=str, default="20260411T115941_isihara_0.0_0.01_8_0.975_5_40.0_1_0")
    parser.add_argument('--validation_load_step_indices', type=int, nargs='+', default=None)
    parser.add_argument('--step', '--dist_step', type=int, default=16, dest='dist_step', help="Step index for local node distribution and mesh contour plots (default: 16)")
    parser.add_argument('--n_sample', type=int, default=128)
    parser.add_argument('--subfolder', type=str, default="fem_validation")

    return parser.parse_args()

if __name__ == "__main__" :
    args = parse_args()
    validation_load_step_indices = args.validation_load_step_indices
    n_sample = args.n_sample
    model_path = args.model_path

    # load result 
    true_data_dir = Path("dataset/precomputed_vfm")

    analysis_dir = Path("validation/coverage_test") 
    extraction_result_dir = Path("extraction/extracted_models") 
    case_name = args.model_path

    if os.path.exists(Path(args.model_path)):
        pred_dir_name = Path(args.model_path)
    else:
        pred_dir_name = analysis_dir / case_name

    if (pred_dir_name / args.subfolder / "fem_distilled_samples.npz").exists():
        pred_dir_name = pred_dir_name / args.subfolder
    elif not (pred_dir_name / "fem_distilled_samples.npz").exists() and (pred_dir_name.parent / args.subfolder / "fem_distilled_samples.npz").exists():
        pred_dir_name = pred_dir_name.parent / args.subfolder
    elif (pred_dir_name / "fem_validation" / "fem_distilled_samples.npz").exists():
        pred_dir_name = pred_dir_name / "fem_validation"
    elif not (pred_dir_name / "fem_distilled_samples.npz").exists() and (pred_dir_name.parent / "fem_validation" / "fem_distilled_samples.npz").exists():
        pred_dir_name = pred_dir_name.parent / "fem_validation"

    save_path = pred_dir_name
    save_path.mkdir(parents=True, exist_ok=True)

    consolidated_file = pred_dir_name / "fem_distilled_samples.npz"

    if os.path.exists(consolidated_file):
        consolidated_data = np.load(consolidated_file, allow_pickle=True)
        max_steps_avail = consolidated_data["u_pred"].shape[1]
    else:
        consolidated_data = None
        max_steps_avail = 9999

    if validation_load_step_indices is None:
        # Search for test_load_steps_indices from ancestor config.yaml / config.json / recipe_config.yaml
        cfg_test_steps = None
        for ancestor in [pred_dir_name, pred_dir_name.parent, pred_dir_name.parent.parent, pred_dir_name.parent.parent.parent, pred_dir_name.parent.parent.parent.parent]:
            for fname in ["config.yaml", "config.json", "recipe_config.yaml"]:
                cfg_p = ancestor / fname
                if cfg_p.exists():
                    try:
                        import yaml
                        with open(cfg_p, "r") as f:
                            d = yaml.safe_load(f)
                        if d and "test_load_steps_indices" in d and d["test_load_steps_indices"]:
                            cfg_test_steps = [int(s) for s in d["test_load_steps_indices"]]
                            break
                    except Exception:
                        pass
            if cfg_test_steps is not None:
                break

        if cfg_test_steps is not None:
            valid_val_step_indices = [s for s in cfg_test_steps if s < max_steps_avail]
            print(f"Using test_load_steps_indices from config: {valid_val_step_indices}")
        elif max_steps_avail <= 3:
            valid_val_step_indices = list(range(max_steps_avail))
        else:
            valid_val_step_indices = list(range(max_steps_avail - 3, max_steps_avail))
    else:
        valid_val_step_indices = [s for s in validation_load_step_indices if s < max_steps_avail]
        if not valid_val_step_indices:
            valid_val_step_indices = [max_steps_avail - 1]
        if len(valid_val_step_indices) > 3:
            valid_val_step_indices = valid_val_step_indices[-3:]

    print(f"Evaluating load step indices (last {len(valid_val_step_indices)} steps): {valid_val_step_indices}")

    if args.dist_step is not None and args.dist_step < max_steps_avail:
        step = args.dist_step
    else:
        step = valid_val_step_indices[-1]
    print(f"Plotting local node distribution and field contours at load step: {step}")

    if consolidated_data is not None:
        u_pred_piola_samples = consolidated_data["u_pred"][:, step]
        mesh_node_coords = consolidated_data["node_coords"]
        mesh_cells = consolidated_data["cells"]
        true_data = consolidated_data
        if "u_true" in consolidated_data:
            u_true = consolidated_data["u_true"][step]
            print(f"Using ground truth displacement (u_true) as reference for plots.")
        elif "u_exp" in consolidated_data:
            u_true = consolidated_data["u_exp"][step]
            print(f"Using experimental displacement (u_exp) as ground truth reference for plots.")
        else:
            gt_data = np.load(pred_dir_name.parent / "gt" / "u_gt.npz")
            u_true = gt_data["u"][step]
        print(f"Loaded {u_pred_piola_samples.shape[0]} consolidated samples from {consolidated_file}")
    else:
        true_data = np.load(pred_dir_name / "gt" / "u_gt.npz")
        u_true = true_data["u"][step]
        files = os.listdir(pred_dir_name / "piola_samples") if os.path.exists(pred_dir_name / "piola_samples") else []
        u_pred_piola_samples = [] 
        for f in files:
            data = np.load(pred_dir_name / "piola_samples" / f)
            u_pred_piola_samples.append(data["u_pred"][step])
        u_pred_piola_samples = jnp.array(u_pred_piola_samples)
        mesh_node_coords = true_data["node_coords"]
        mesh_cells = true_data["cells"]

    pt_files = os.listdir(pred_dir_name / "piola_traction_samples") if os.path.exists(pred_dir_name / "piola_traction_samples") else []
    u_pred_piola_traction_samples = []
    for f in pt_files:
        data = np.load(pred_dir_name / "piola_traction_samples" / f)
        u_pred_piola_traction_samples.append(data["u_pred"][step])
    if len(u_pred_piola_traction_samples) > 0:
        u_pred_piola_traction_samples = jnp.array(u_pred_piola_traction_samples)
    else:
        u_pred_piola_traction_samples = None

    targets = np.array([
        [0.0707, 0.0707],
        [0.25, 0.75],
        [0.6, 0.4],
        [1, 1]
    ])
    node_indices = []
    for target in targets:
        dist = np.linalg.norm(mesh_node_coords - target, axis=1)
        node_indices.append(np.argmin(dist).item())

    print(f"Closest node indices: {node_indices}")
    node_type = true_data["node_type"]

    ref_u_all = true_data["u_true"] if "u_true" in true_data else (true_data["u_exp"] if "u_exp" in true_data else true_data["u"])
    u_true_val = ref_u_all[valid_val_step_indices]

    if os.path.exists(consolidated_file):
        u_pred_piola_samples_val = consolidated_data["u_pred"][:, valid_val_step_indices]
    else:
        files = os.listdir(pred_dir_name / "piola_samples") if os.path.exists(pred_dir_name / "piola_samples") else []
        u_pred_piola_samples_val = [] 
        for f in files:
            data = np.load(pred_dir_name / "piola_samples" / f)
            u_pred_piola_samples_val.append(data["u_pred"][valid_val_step_indices])
        u_pred_piola_samples_val = jnp.array(u_pred_piola_samples_val)
    p_val_shape = u_pred_piola_samples_val.shape
    u_pred_piola_samples_val_flat = u_pred_piola_samples_val.reshape(p_val_shape[0], -1, 2)

    u_true_val_flat = u_true_val.reshape(-1, 2)
    u_p_lower_bound = np.quantile(u_pred_piola_samples_val_flat, 0.025, axis=0)
    u_p_upper_bound = np.quantile(u_pred_piola_samples_val_flat, 0.975, axis=0)
    u_p_mean = np.mean(u_pred_piola_samples_val_flat, axis=0)

    # Detect control_mode and stress_mode
    ctrl_mode = "force"
    stress_mode = "plane_strain"
    if consolidated_data is not None:
        if "control_mode" in consolidated_data:
            ctrl_mode = str(consolidated_data["control_mode"])
        if "stress_mode" in consolidated_data:
            stress_mode = str(consolidated_data["stress_mode"])
    elif true_data is not None:
        if "control_mode" in true_data:
            ctrl_mode = str(true_data["control_mode"])
        if "stress_mode" in true_data:
            stress_mode = str(true_data["stress_mode"])
    mode_display = f"{ctrl_mode.capitalize()} Control, {stress_mode.replace('_', ' ').capitalize()}"

    # 1x4 Consolidated Displacement Analysis Plot
    plot_disp_field(
        mesh_node_coords, mesh_cells, u_true, u_pred_piola_samples.mean(axis=0),
        u_true_val_flat, u_p_mean, u_p_lower_bound, u_p_upper_bound,
        save_path,
        mode_str=mode_display
    )

    plot_node_distributions(u_true, u_pred_piola_samples, u_pred_piola_traction_samples, node_indices, save_path)

    if len(pt_files) > 0:
        u_pred_piola_traction_samples_val = []
        for f in pt_files:
            data = np.load(pred_dir_name / "piola_traction_samples" / f)
            u_pred_piola_traction_samples_val.append(data["u_pred"][valid_val_step_indices])
        u_pred_piola_traction_samples_val = jnp.array(u_pred_piola_traction_samples_val)
        pt_val_shape = u_pred_piola_traction_samples_val.shape
        u_pred_piola_traction_samples_val_flat = u_pred_piola_traction_samples_val.reshape(pt_val_shape[0], -1, 2)

        u_pt_lower_bound = np.quantile(u_pred_piola_traction_samples_val_flat, 0.025, axis=0)
        u_pt_upper_bound = np.quantile(u_pred_piola_traction_samples_val_flat, 0.975, axis=0)
        u_pt_mean = np.mean(u_pred_piola_traction_samples_val_flat, axis=0)

        plot_disp_r2_coverage(u_true_val_flat, u_pt_mean, u_pt_lower_bound, u_pt_upper_bound, save_path, suffix ="_piola_traction")

    plot_disp_r2_coverage(u_true_val_flat, u_p_mean, u_p_lower_bound, u_p_upper_bound, save_path, suffix ="_piola")