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

from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import os
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib.ticker as ticker

def plot_comprehensive_analysis(u_true, u_pred_samples, node_type, node_to_plot, save_path):
    """
    Generates and saves two separate 2x2 figures (X and Y directions).
    Includes statistical text annotations and a zero-line for error distributions.
    """
    # Filter for free nodes
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
        ax1.fill_between(node_indices, 0, u_std_per_node, color=main_color, alpha=0.15, label='$\sigma$ per Node')
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
    plt.savefig(save_file, bbox_inches='tight', transparent=True)
    # print(f"Transposed plot saved to: {save_file}")
    # plt.show()
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
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_disp_field(node_coords, cells, u_true, u_pred_mean, u_pred_std, node_indices, save_path):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "text.usetex": False
    })

    # --- Data Preparation ---
    node_indices = np.array(node_indices)
    coords_true = node_coords + u_true
    coords_pred = node_coords + u_pred_mean
    
    def get_mag(u): return np.linalg.norm(u, axis=1)
    mag_true = get_mag(u_true)
    mag_pred = get_mag(u_pred_mean)
    error = np.linalg.norm(u_true - u_pred_mean, axis=1)/mag_true * 100
    mag_std = get_mag(u_pred_std) if u_pred_std.ndim > 1 else u_pred_std

    marker_coords_true = coords_true[node_indices]
    marker_coords_pred = coords_pred[node_indices]

    fig, axes = plt.subplots(2, 2, figsize=(12*1.5, 14*1.5)) # Slightly wider to accommodate colorbars
    # plt.suptitle('Deformed Field: Accuracy & Uncertainty', fontsize=20)

    # --- Helper Function for Labels ---
    def add_markers_with_labels(ax, coords, indices):
        ax.scatter(coords[:, 0], coords[:, 1], 
                   color='red', marker='x', s=60*1.5, linewidths=1.5, 
                   label='Probe Nodes', zorder=8)
        for i, idx in enumerate(indices):
                    ax.annotate(f'ID: {idx}', 
                                (coords[i, 0], coords[i, 1]),
                                textcoords="offset points", 
                                xytext=(-60, 10),        # Slightly increased vertical offset for better clearance
                                fontsize=28, 
                                fontweight='bold',
                                color='red',
                                zorder=7,
                                # --- Background Box Styling ---
                                bbox=dict(
                                    boxstyle='round,pad=0.3', 
                                    facecolor='white', 
                                    edgecolor='red', 
                                    alpha=0.8,
                                    linewidth=1
                                ))

    # --- Helper Function for Colorbars ---
    def add_colorbar(im, ax, label):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        
        # cbar = fig.colorbar(im, cax=cax, label=label)
        cbar = fig.colorbar(im, cax=cax)
        
        # 1. Set the main colorbar title (label) and its size
        cbar.set_label(label, size=36, weight='bold') # Adjust '16' as needed
        
        # 2. Set the size of the tick values (the numbers)
        cbar.ax.tick_params(labelsize=36) # Adjust '14' as needed
        cbar.locator = ticker.MaxNLocator(nbins=4)
        cbar.update_ticks()

    # 1,1: True Material
    tri_true = tri.Triangulation(coords_true[:, 0], coords_true[:, 1], cells)
    im1 = axes[0, 0].tripcolor(tri_true, mag_true, cmap='Blues')
    add_markers_with_labels(axes[0, 0], marker_coords_true, node_indices)
    # axes[0, 0].set_title('True Material Model $\|\mathbf{u_{true}}\|$')
    add_colorbar(im1, axes[0, 0], "$\|\mathbf{u_{true}}\|$")

    # 1,2: Predicted Material
    tri_pred = tri.Triangulation(coords_pred[:, 0], coords_pred[:, 1], cells)
    im2 = axes[0, 1].tripcolor(tri_pred, mag_pred, cmap='Blues')
    add_markers_with_labels(axes[0, 1], marker_coords_pred, node_indices)
    # axes[0, 1].set_title('Predicted Material Model $\|\mathbf{u_{pred}}\|$')
    add_colorbar(im2, axes[0, 1], "$\|\mathbf{u_{pred}}\|$")

    # 2,1: Nodal error
    im3 = axes[1, 0].tripcolor(tri_pred, error, cmap='inferno')
    add_markers_with_labels(axes[1, 0], marker_coords_pred, node_indices)
    # axes[1, 0].set_title(r'$||\mthbf{u_{true}} - \mathbf{u_{pred}}||$')
    add_colorbar(im3, axes[1, 0], r"$\% Error$")

    # 2,2: Uncertainty
    im4 = axes[1, 1].tripcolor(tri_pred, mag_std, cmap='magma')
    add_markers_with_labels(axes[1, 1], marker_coords_pred, node_indices)
    # axes[1, 1].set_title(r'Uncertainty ($\sigma_u$)')
    add_colorbar(im4, axes[1, 1], "$\sigma_{\|\mathbf{u_{pred}}\|}$")

    # Standardize labels
    for ax in axes.flat:
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # for ax in axes.flat:
        # Remove axis labels
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Remove tick marks and tick labels (values)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        # Ensure the grid is off so it doesn't show through the transparency
        # ax.grid(False)
        # Optional: if you want to remove the frame/box as well, uncomment the line below
        ax.axis('off') 
        
        # ax.set_aspect('equal')
        ax.set_aspect('equal')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    # plt.tight_layout(rect=[0, 0, 1, 1])
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "displacement_analysis.pdf"), bbox_inches='tight')


import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import r2_score

def plot_disp_r2_coverage(u_true, u_pred_med, u_pred_lower, u_pred_upper, save_path, suffix="_"):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 28,
        "axes.titlesize": 32,
        "axes.labelsize": 32,
        "legend.fontsize": 28,
        "xtick.labelsize": 28,
        "ytick.labelsize": 28,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "text.usetex": False
    })

    fig, ax = plt.subplots(figsize=(15, 9))
    
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
    # Relative errors for Matplotlib yerr
    ux_err = [ux_med - ux_lower, ux_upper - ux_med]
    uy_err = [uy_med - uy_lower, uy_upper - uy_med]

    # 1. Plot Displacement X (Blue, 'x' marker)
    ax.errorbar(ux_true, ux_med, yerr=ux_err, fmt='x', color='blue', ecolor='blue', 
                alpha=0.3, label=f'$u_x$ (Cov: {cov_x:.1f}%)', markersize=6, capsize=0)
    
    # 2. Plot Displacement Y (Red, 'o' marker)
    ax.errorbar(uy_true, uy_med, yerr=uy_err, fmt='o', color='red', ecolor='red', 
                alpha=0.3, label=f'$u_y$ (Cov: {cov_y:.1f}%)', markersize=4, capsize=0)

    # 3. Identity line (Black dashed)
    all_vals = np.concatenate([u_true.flatten(), u_pred_med.flatten()])
    limits = [all_vals.min(), all_vals.max()]
    ax.plot(limits, limits, 'k--', linewidth=2, label='Isoline', zorder=5)

    # --- Annotation Box ---
    stats_text = (f'Estimated Coverage $X$: {cov_x:.1f}%\n'
                  f'Estimated Coverage $Y$: {cov_y:.1f}%\n'
                  f'$R^2_X$: {r2_x:.4f}\n'
                  f'$R^2_Y$: {r2_y:.4f}')
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=28, 
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Formatting
    ax.set_xlabel('$u_{gt}$')
    ax.set_ylabel('$u_{pred}$')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', frameon=True, fontsize=28)
    
    # Ticks limit
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

    plt.tight_layout()
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, f"disp_r2_coverage_xy_{suffix}.pdf")
        plt.savefig(save_file, bbox_inches='tight')
        print(f"Plot saved to: {save_file}")
    
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="Isihara Model Dataset and Training Configuration")

    # Dataset & Model Config
    parser.add_argument('--model_path', type=str, default="20260411T115941_isihara_0.0_0.01_8_0.975_5_40.0_1_0")
    parser.add_argument('--validation_load_step_indices', type=int, nargs='+', default=[2, 4, 6, 8])
    parser.add_argument('--n_sample', type=int, default=128)


    return parser.parse_args()

if __name__ == "__main__" :
    # plt.rcParams.update({
    # "font.family": "serif",
    # "font.serif": ["Times New Roman", "DejaVu Serif"], # Falls back to DejaVu if Times isn't found
    # "font.size": 14,                # Base font size
    # "axes.titlesize": 22,           # Subplot titles
    # "axes.labelsize": 20,           # X and Y labels
    # "legend.fontsize": 16,          # Legend text
    # "xtick.labelsize": 13,          # Axis tick numbers
    # "ytick.labelsize": 13,
    # "figure.dpi": 600,              # High resolution for the screen and save
    # "savefig.dpi": 600,             # Ensures saved file is high quality
    # "text.usetex": False            # Set to True only if you have a full LaTeX install
    # })
    args = parse_args()
    # read file from coverage/{mat_model}_{disp_noise}_{load_noise}
    # material_model_name = "isihara"
    validation_load_step_indices = args.validation_load_step_indices
    print(validation_load_step_indices)
    n_sample = args.n_sample
    model_path = args.model_path

    # load result 

    true_data_dir = Path("dataset/precomputed_vfm")

    analysis_dir = Path("validation/coverage_test") 
    extraction_result_dir = Path("extraction/extracted_models") 
    case_name = args.model_path
    # precomputed_vfm_name = f"{case_name.split('_')[1]}_{case_name.split('_')[2]}_{case_name.split('_')[3]}_{case_name.split('_')[4]}_{case_name.split('_')[5]}"

    if os.path.exists(Path(args.model_path)):
        pred_dir_name = Path(args.model_path)
    else:
        pred_dir_name = analysis_dir / case_name

    if (pred_dir_name / "fem_validation" / "fem_distilled_samples.npz").exists():
        pred_dir_name = pred_dir_name / "fem_validation"
    elif not (pred_dir_name / "fem_distilled_samples.npz").exists() and (pred_dir_name.parent / "fem_validation" / "fem_distilled_samples.npz").exists():
        pred_dir_name = pred_dir_name.parent / "fem_validation"

    save_path = pred_dir_name
    save_path.mkdir(parents=True, exist_ok=True)

    step = validation_load_step_indices[-1]
    consolidated_file = pred_dir_name / "fem_distilled_samples.npz"

    if os.path.exists(consolidated_file):
        consolidated_data = np.load(consolidated_file, allow_pickle=True)
        u_pred_piola_samples = consolidated_data["u_pred"][:, step]
        mesh_node_coords = consolidated_data["node_coords"]
        mesh_cells = consolidated_data["cells"]
        true_data = consolidated_data
        if "u_exp" in consolidated_data:
            u_true = consolidated_data["u_exp"][step]
            print(f"Using experimental displacement (u_exp) as ground truth reference for plots.")
        elif "u_true" in consolidated_data:
            u_true = consolidated_data["u_true"][step]
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

    # err = u_samples - u_true[None, :, :]
    import numpy as np

    # Define target locations on the r = 0.1 circle
    targets = np.array([
        # [0.0866, 0.05],
        [0.0707, 0.0707],
        # [0.05, 0.0866],
        [0.25, 0.75],
        [0.6, 0.4],
        # [0.75, 0.25],
        [1, 1]


    ])
    # u_samples = u_true_samples
    # Find the indices of the closest nodes in your mesh
    node_indices = []
    for target in targets:
        dist = np.linalg.norm(mesh_node_coords - target, axis=1)
        node_indices.append(np.argmin(dist).item())

    print(f"Closest node indices: {node_indices}")
    node_type = true_data["node_type"]
    plot_disp_field(mesh_node_coords, mesh_cells, u_true, u_pred_piola_samples.mean(axis=0), u_pred_piola_samples.std(axis=0), node_indices, save_path)

    plot_node_distributions(u_true, u_pred_piola_samples, u_pred_piola_traction_samples, node_indices, save_path)
    # plot_disp_r2_coverage(u_true, u_pred_samples.mean(axis=0), u_pred_samples.std(axis=0), save_path)

    # true_data = np.load(true_data_dir / f"{precomputed_vfm_name}.npz")
    u_true_val = (true_data["u_exp"] if "u_exp" in true_data else (true_data["u_true"] if "u_true" in true_data else true_data["u"]))[validation_load_step_indices]

    if os.path.exists(consolidated_file):
        u_pred_piola_samples_val = consolidated_data["u_pred"][:, validation_load_step_indices]
    else:
        files = os.listdir(pred_dir_name / "piola_samples") if os.path.exists(pred_dir_name / "piola_samples") else []
        u_pred_piola_samples_val = [] 
        for f in files:
            data = np.load(pred_dir_name / "piola_samples" / f)
            u_pred_piola_samples_val.append(data["u_pred"][validation_load_step_indices])
        u_pred_piola_samples_val = jnp.array(u_pred_piola_samples_val)
    p_val_shape = u_pred_piola_samples_val.shape
    u_pred_piola_samples_val_flat = u_pred_piola_samples_val.reshape(p_val_shape[0], -1, 2)
    if len(pt_files) > 0:
        u_pred_piola_traction_samples_val = []
        for f in pt_files:
            data = np.load(pred_dir_name / "piola_traction_samples" / f)
            u_pred_piola_traction_samples_val.append(data["u_pred"][validation_load_step_indices])
        u_pred_piola_traction_samples_val = jnp.array(u_pred_piola_traction_samples_val)
        pt_val_shape = u_pred_piola_traction_samples_val.shape

        u_pred_piola_traction_samples_val_flat = u_pred_piola_traction_samples_val.reshape(pt_val_shape[0], -1, 2)
        u_true_val_flat = u_true_val.reshape(-1, 2)

        u_pt_lower_bound = np.quantile(u_pred_piola_traction_samples_val_flat, 0.025, axis=0)
        u_pt_upper_bound = np.quantile(u_pred_piola_traction_samples_val_flat, 0.975, axis=0)
        u_pt_mean = np.mean(u_pred_piola_traction_samples_val_flat, axis=0)

        plot_disp_r2_coverage(u_true_val_flat, u_pt_mean, u_pt_lower_bound, u_pt_upper_bound, save_path, suffix ="_piola_traction")
    else:
        u_true_val_flat = u_true_val.reshape(-1, 2)


    u_p_lower_bound = np.quantile(u_pred_piola_samples_val_flat, 0.025, axis=0)
    u_p_upper_bound = np.quantile(u_pred_piola_samples_val_flat, 0.975, axis=0)
    u_p_mean = np.mean(u_pred_piola_samples_val_flat, axis=0)
    plot_disp_r2_coverage(u_true_val_flat, u_p_mean, u_p_lower_bound, u_p_upper_bound, save_path, suffix ="_piola")
    # for n_idx in node_indices :
    #     plot_comprehensive_analysis(u_true, u_pred_samples, node_type, n_idx, save_path)
    pass