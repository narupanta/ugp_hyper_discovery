import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_dic_displacement_plot():
    dataset_path = 'dataset/experimental/18617429'
    hdf5_file = os.path.join(dataset_path, 'TPS_2.hdf5')
    
    if not os.path.exists(hdf5_file):
        raise FileNotFoundError(f"Could not find DIC dataset at {hdf5_file}")
        
    print(f"Loading experimental dataset from {hdf5_file}...")
    with h5py.File(hdf5_file, 'r') as f:
        nodes = f['region/nodes'][()]
        u_group = f['output/u']
        load_step_keys = sorted(list(u_group.keys()))
        
        # Select three informative load steps: early (~25%), middle (~60%), and final (100%)
        step_indices = [
            int(len(load_step_keys) * 0.25),
            int(len(load_step_keys) * 0.60),
            len(load_step_keys) - 1
        ]
        selected_keys = [load_step_keys[i] for i in step_indices]
        
        # Load displacement fields for selected steps
        u_data = [u_group[key][()] for key in selected_keys]
        
    # Extract undeformed coordinates (2D: X and Y)
    X0 = nodes[:, 0]
    Y0 = nodes[:, 1]
    
    print("Generating publication-quality displacement visualization...")
    fig, axes = plt.subplots(3, 3, figsize=(16, 16), dpi=200)
    fig.suptitle('Experimental DIC Displacement Field Analysis (Dataset 18617429)', fontsize=20, fontweight='bold', y=0.97)
    
    # Configure rows: Ux, Uy, and Magnitude
    row_labels = [
        r'Horizontal Displacement $u_x$ (px / mm)', 
        r'Vertical Displacement $u_y$ (px / mm)', 
        r'Total Displacement Magnitude $\|u\|$'
    ]
    col_cmaps = ['coolwarm', 'viridis', 'plasma']
    
    for col_idx, (step_idx, key, u) in enumerate(zip(step_indices, selected_keys, u_data)):
        # Deformed coordinates
        X_def = X0 + u[:, 0]
        Y_def = Y0 + u[:, 1]
        
        ux = u[:, 0]
        uy = u[:, 1]
        umag = np.sqrt(ux**2 + uy**2)
        
        fields = [ux, uy, umag]
        cmaps = ['coolwarm', 'coolwarm', 'viridis']
        
        for row_idx, (field, cmap) in enumerate(zip(fields, cmaps)):
            ax = axes[row_idx, col_idx]
            
            # Scatter plot for full resolution without boundary triangulation artifacts
            sc = ax.scatter(X_def, Y_def, c=field, cmap=cmap, s=1.5, alpha=0.9, edgecolors='none')
            
            # Plot undeformed outline/bounding silhouette as subtle grey reference
            ax.scatter(X0, Y0, c='gray', s=0.2, alpha=0.15, label='Undeformed Reference' if row_idx==0 and col_idx==0 else "")
            
            # Formatting and labeling
            cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=10)
            
            if row_idx == 0:
                ax.set_title(f'Step {int(key)} ({int((step_idx+1)/len(load_step_keys)*100)}% Load)', fontsize=14, fontweight='bold', pad=10)
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx] + '\n\n$Y$ Coordinate', fontsize=12, fontweight='bold')
            else:
                ax.set_ylabel('$Y$ Coordinate', fontsize=11)
                
            ax.set_xlabel('$X$ Coordinate', fontsize=11)
            ax.set_aspect('equal', 'box')
            ax.grid(True, linestyle='--', alpha=0.3)
            
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    
    os.makedirs('plots/figures', exist_ok=True)
    out_path = 'plots/figures/experimental_displacement_field_18617429.pdf'
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Successfully generated plot and saved to {out_path}!")

if __name__ == "__main__":
    generate_dic_displacement_plot()
