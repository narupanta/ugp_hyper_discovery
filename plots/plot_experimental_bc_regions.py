import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import os

def identify_boundary_nodes(nodes, max_edge_len=15.0):
    """
    Identifies perimeter nodes of a planar mesh by filtering long edges
    from a 2D Delaunay triangulation.
    """
    pts = nodes[:, :2]
    tri = Delaunay(pts)
    
    # Compute side lengths of each triangle to prune artificial bridges across concave notches
    t_pts = pts[tri.simplices]
    l1 = np.linalg.norm(t_pts[:, 0] - t_pts[:, 1], axis=-1)
    l2 = np.linalg.norm(t_pts[:, 1] - t_pts[:, 2], axis=-1)
    l3 = np.linalg.norm(t_pts[:, 2] - t_pts[:, 0], axis=-1)
    max_l = np.maximum(np.maximum(l1, l2), l3)
    
    valid_tris = tri.simplices[max_l < max_edge_len]
    
    # Extract all edges from valid triangles
    edges = np.vstack([
        valid_tris[:, [0, 1]],
        valid_tris[:, [1, 2]],
        valid_tris[:, [2, 0]]
    ])
    edges = np.sort(edges, axis=1)
    
    # Boundary edges appear exactly once in a 2D manifold mesh
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    
    boundary_nodes = np.unique(boundary_edges)
    return boundary_nodes, valid_tris

def generate_bc_visualization():
    dataset_path = 'dataset/experimental/18617429'
    hdf5_file = os.path.join(dataset_path, 'TPS_2.hdf5')
    
    if not os.path.exists(hdf5_file):
        raise FileNotFoundError(f"Could not find DIC dataset at {hdf5_file}")
        
    print(f"Loading experimental dataset from {hdf5_file}...")
    with h5py.File(hdf5_file, 'r') as f:
        nodes = f['region/nodes'][()]
        u_group = f['output/u']
        last_step_key = sorted(list(u_group.keys()))[-1]
        u_last = u_group[last_step_key][()]
        
    X = nodes[:, 0]
    Y = nodes[:, 1]
    
    print("Detecting external perimeter and identifying VFM boundary condition zones...")
    perimeter_indices, valid_tris = identify_boundary_nodes(nodes, max_edge_len=15.0)
    
    # Identify specific cropped boundaries based on geometric coordinates
    top_mask = (Y > Y.max() - 2.0)
    bottom_mask = (Y < Y.min() + 2.0)
    left_mask = (X < X.min() + 2.0)
    
    # Create masks for full node array
    is_top = top_mask
    is_bottom = bottom_mask
    is_left = left_mask
    
    # Free exterior border: nodes that lie on perimeter but NOT on any of the cropped clamp cuts
    is_perimeter = np.zeros(len(nodes), dtype=bool)
    is_perimeter[perimeter_indices] = True
    is_free_perimeter = is_perimeter & ~is_top & ~is_bottom & ~is_left
    
    # Interior free DOFs: not on the perimeter at all
    is_interior = ~is_perimeter
    
    print(f"Node Classification Breakdown:\n"
          f"  - Top Neumann Loaded Nodes: {np.sum(is_top)}\n"
          f"  - Bottom Dirichlet Clamp Nodes: {np.sum(is_bottom)}\n"
          f"  - Left Dirichlet Clamp Nodes: {np.sum(is_left)}\n"
          f"  - Free Curved Perimeter Nodes: {np.sum(is_free_perimeter)}\n"
          f"  - Interior Free DOFs: {np.sum(is_interior)}\n"
          f"  - Total Nodes Evaluated: {len(nodes)}")
          
    # Setup plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), dpi=200)
    fig.suptitle('VFM Boundary Condition Identification for Cropped DIC Domain (Dataset 18617429)', 
                 fontsize=18, fontweight='bold', y=0.96)
                 
    # --- Panel 1: Undeformed Reference with VFM BC Classification ---
    ax1.scatter(X[is_interior], Y[is_interior], c='#C0C0C0', s=1.0, alpha=0.5, 
                label=r'Interior Free DOFs ($\mathbf{f}_{int} = \mathbf{0}$)')
    ax1.scatter(X[is_free_perimeter], Y[is_free_perimeter], c='#2CA02C', s=12, alpha=0.9, zorder=3,
                label=r'Free External Borders ($\mathbf{t} = \mathbf{0} \rightarrow \mathbf{f}_{int} = \mathbf{0}$)')
    ax1.scatter(X[is_left], Y[is_left], c='#9467BD', s=25, alpha=1.0, zorder=4, marker='o',
                label=r'Left Cropped Clamp (Dirichlet: Reaction boundary)')
    ax1.scatter(X[is_bottom], Y[is_bottom], c='#1F77B4', s=25, alpha=1.0, zorder=4, marker='o',
                label=r'Bottom Cropped Clamp (Dirichlet: Reaction boundary)')
    ax1.scatter(X[is_top], Y[is_top], c='#D62728', s=30, alpha=1.0, zorder=5, marker='^',
                label=r'Top Cropped Grip (Neumann: Match Load Cell $F_{meas}$)')
                
    # Annotations on Panel 1
    ax1.annotate('Top Grip (Pull Direction)\nMatch total load-cell force via VFM', 
                 xy=(X[is_top].mean(), Y[is_top].mean()), xytext=(1350, 2050),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
                 fontsize=11, fontweight='bold', bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="red", lw=1.5))
                 
    ax1.annotate('Left Clamp (Fixed Grip)\nExcluded from local residual;\nreaction equilibrium constraint', 
                 xy=(X[is_left].mean(), Y[is_left].mean()), xytext=(X[is_left].mean()+60, Y[is_left].mean()-350),
                 arrowprops=dict(facecolor='purple', shrink=0.05, width=2, headwidth=8),
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="purple", lw=1.5))
                 
    ax1.annotate('Bottom Clamp (Fixed Grip)\nExcluded from local residual;\nreaction equilibrium constraint', 
                 xy=(X[is_bottom].mean(), Y[is_bottom].mean()), xytext=(X[is_bottom].mean()+120, Y[is_bottom].mean()-180),
                 arrowprops=dict(facecolor='blue', shrink=0.05, width=2, headwidth=8),
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="blue", lw=1.5))
                 
    ax1.set_xlim(250, 1850)
    ax1.set_ylim(450, 2500)
    ax1.set_title('Undeformed DIC Domain: Boundary Classification Map', fontsize=15, fontweight='bold', pad=15)
    ax1.set_xlabel('$X$ Coordinate (px/mm)', fontsize=13)
    ax1.set_ylabel('$Y$ Coordinate (px/mm)', fontsize=13)
    ax1.set_aspect('equal', 'box')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.95, edgecolor='black')
    
    # --- Panel 2: Deformed Configuration (100% Load) with BC Overlay ---
    X_def = X + u_last[:, 0]
    Y_def = Y + u_last[:, 1]
    umag = np.sqrt(u_last[:, 0]**2 + u_last[:, 1]**2)
    
    sc = ax2.scatter(X_def, Y_def, c=umag, cmap='plasma', s=2.5, alpha=0.85, zorder=1)
    cbar = plt.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Total Displacement Magnitude $\|u\|$', fontsize=12, fontweight='bold', labelpad=10)
    
    # Overlay deformed boundaries in prominent distinct colors
    ax2.scatter(X_def[is_free_perimeter], Y_def[is_free_perimeter], c='#2E7D32', s=8, alpha=0.7, zorder=3, label='Free External Border')
    ax2.scatter(X_def[is_left], Y_def[is_left], c='#00E676', edgecolors='black', s=45, lw=0.8, alpha=1.0, zorder=4, label='Deformed Left Clamp (DBC)')
    ax2.scatter(X_def[is_bottom], Y_def[is_bottom], c='#00E5FF', edgecolors='black', s=45, lw=0.8, alpha=1.0, zorder=4, label='Deformed Bottom Clamp (DBC)')
    ax2.scatter(X_def[is_top], Y_def[is_top], c='#FF1744', edgecolors='black', s=55, lw=0.8, alpha=1.0, zorder=5, marker='^', label='Deformed Top Pull Grip (NBC)')
    
    # Reference shadow of undeformed outline
    ax2.scatter(X[is_perimeter], Y[is_perimeter], c='black', s=3, alpha=0.2, label='Undeformed Outline Ref')
    
    ax2.set_title(f'Deformed Configuration at Peak Load (Step {int(last_step_key)})\nShowing Boundary Tracking & Hyperelastic Stretch', fontsize=15, fontweight='bold', pad=15)
    ax2.set_xlabel('Deformed $X$ Coordinate ($X + u_x$)', fontsize=13)
    ax2.set_ylabel('Deformed $Y$ Coordinate ($Y + u_y$)', fontsize=13)
    ax2.set_aspect('equal', 'box')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(loc='lower right', fontsize=11, framealpha=0.9, edgecolor='black')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    
    os.makedirs('plots/figures', exist_ok=True)
    out_path = 'plots/figures/experimental_bc_identification_18617429.pdf'
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Successfully generated BC identification visualization and saved to {out_path}!")

if __name__ == "__main__":
    generate_bc_visualization()
