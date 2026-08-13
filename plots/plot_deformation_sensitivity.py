import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def extract_gamma(F, mode_idx):
    """Extract stretch measure gamma from deformation gradient F based on mode index."""
    if mode_idx in [0, 1, 2]:  # UT, EBT, PS
        return F[0, 0] - 1.0
    elif mode_idx in [3, 4]:  # UC, EBC
        return 1.0 / F[0, 0] - 1.0
    elif mode_idx == 5:  # SS
        return F[0, 1]
    return 0.0

def main():
    parser = argparse.ArgumentParser(description="Plot deformation-dependent Sobol indices")
    parser.add_argument("--distilled_dir", type=str, required=True, help="Path to the distilled model directory")
    args = parser.parse_args()
    
    csv_path = os.path.join(args.distilled_dir, "output", "sensitivities", "total_sobol_indices_output_0.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Exclude non-parameter columns
    exclude_cols = ['test cases', 'Unnamed: 0']
    params = [c for c in df.columns if c not in exclude_cols and not c.startswith('Unnamed')]
    
    # Find the source extraction directory to load f3x3.npy
    source_txt = os.path.join(args.distilled_dir, "source_extraction_dir.txt")
    if not os.path.exists(source_txt):
        print(f"Error: {source_txt} not found. Cannot determine gamma stretch.")
        return
        
    with open(source_txt, 'r') as f:
        source_dir = f.read().strip()
        
    # Find f3x3.npy
    f3x3_path = None
    for root, dirs, files in os.walk(source_dir):
        if "f3x3.npy" in files:
            f3x3_path = os.path.join(root, "f3x3.npy")
            break
            
    if f3x3_path is None:
        print(f"Error: f3x3.npy not found in {source_dir}")
        return
        
    f3x3 = np.load(f3x3_path)
    num_orig_points = f3x3.shape[0]
    
    # UQInModelDiscovery evaluates all points passed to it except for the origins (F=I).
    # distill_uqmodeldisc.py masks out ALL zero-strain origins by setting their test case to 2 (biaxial tension)
    # which is not in the list of relevant test cases (0, 1, 3).
    keep_mask = np.ones(num_orig_points, dtype=bool)
    identity_matrix = np.eye(3)
    is_zero_strain = np.max(np.max(np.abs(f3x3 - identity_matrix), axis=2), axis=1) < 1e-6
    keep_mask[is_zero_strain] = False
    
    f3x3_kept = f3x3[keep_mask]
    
    if len(f3x3_kept) != len(df):
        print(f"Warning: Number of remaining f3x3 points ({len(f3x3_kept)}) does not match CSV rows ({len(df)})!")
        # We'll just truncate to the minimum to avoid crashing
        min_len = min(len(f3x3_kept), len(df))
        f3x3_kept = f3x3_kept[:min_len]
        df = df.iloc[:min_len]
        
    # Reconstruct the mode labels and gamma values
    mode_names = ["Uniaxial Tension", "Equibiaxial Tension", "Pure Shear", 
                  "Uniaxial Compression", "Equibiaxial Compression", "Simple Shear"]
    
    chunk_size = max(1, num_orig_points // 6)
    
    orig_indices_kept = np.arange(num_orig_points)[keep_mask][:len(df)]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for mode_idx in range(6):
        # Find which rows in the CSV belong to this mode
        row_mask = (orig_indices_kept // chunk_size) == mode_idx
        df_mode = df[row_mask]
        f3x3_mode = f3x3_kept[row_mask]
        
        if len(df_mode) == 0:
            continue
            
        gammas = [extract_gamma(F, mode_idx) for F in f3x3_mode]
        
        ax = axes[mode_idx]
        
        plotted_params = 0
        for param in params:
            y_vals = df_mode[param].values
            if np.max(np.abs(y_vals)) > 1e-4:
                ax.plot(gammas, y_vals, label=param, linewidth=2.5, alpha=0.85, marker='x', markersize=6)
                plotted_params += 1
                
        if plotted_params == 0:
            for param in params:
                ax.plot(gammas, df_mode[param].values, label=param, linewidth=1, alpha=0.5, marker='x', markersize=4)
                
        ax.set_ylim(bottom=0.0, top=1.05)
        ax.set_title(f"({mode_idx + 1}) {mode_names[mode_idx]}", fontsize=14, fontweight='bold')
        ax.set_xlabel(r"Stretch Measure ($\gamma$)", fontsize=12, fontweight='bold')
        if mode_idx % 3 == 0:
            ax.set_ylabel("Total-Order Sobol Index ($S_T$)", fontsize=12, fontweight='bold')
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    # Use a single legend for the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(params)//2 + 1, fontsize=12, title="Parameters", title_fontsize=14)
    
    plt.tight_layout()
    out_path = os.path.join(args.distilled_dir, "deformation_sensitivity_modes.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Plot saved to: {out_path}")

if __name__ == "__main__":
    main()
