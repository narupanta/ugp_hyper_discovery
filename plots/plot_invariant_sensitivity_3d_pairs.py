import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compute_invariants(F):
    """Compute I1_bar, I2_bar, J for an array of deformation gradients."""
    J = np.linalg.det(F)
    F_T = np.transpose(F, (0, 2, 1))
    C = np.einsum('nij,njk->nik', F_T, F)
    J_safe = np.clip(J, 1e-8, 1e8)
    C_bar = C / (J_safe**(2/3))[:, None, None]
    
    I1_bar = np.trace(C_bar, axis1=1, axis2=2)
    C_bar_sq = np.einsum('nij,njk->nik', C_bar, C_bar)
    I2_bar = 0.5 * (I1_bar**2 - np.trace(C_bar_sq, axis1=1, axis2=2))
    
    return I1_bar, I2_bar, J

def main():
    parser = argparse.ArgumentParser(description="Plot invariant-dependent Sobol indices in 3D pairs")
    parser.add_argument("--distilled_dir", type=str, required=True, help="Path to the distilled model directory")
    parser.add_argument("--active_params", type=str, default=None, help="Comma-separated list of active parameters")
    parser.add_argument("--component", type=str, default="dev", choices=["dev", "vol"], help="Component for split model")
    parser.add_argument("--distill_target", type=str, default="sef", choices=["sef", "sef_stress", "sef_cauchy", "sef_split"])
    args = parser.parse_args()
    
    if args.distill_target == "sef_split":
        sens_dir_name = f"{args.component}_sensitivities"
        source_txt_name = f"{args.component}_source_extraction_dir.txt"
        img_prefix = f"{args.component}_"
    else:
        sens_dir_name = "sensitivities"
        source_txt_name = "source_extraction_dir.txt"
        img_prefix = ""

    csv_path = os.path.join(args.distilled_dir, "output", sens_dir_name, "total_sobol_indices_output_0.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(args.distilled_dir, sens_dir_name, "total_sobol_indices_output_0.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    exclude_cols = ['test cases', 'Unnamed: 0']
    params = [c for c in df.columns if c not in exclude_cols and not c.startswith('Unnamed')]
    
    source_txt = os.path.join(args.distilled_dir, source_txt_name)
    if not os.path.exists(source_txt):
        print(f"Error: {source_txt} not found.")
        return
        
    with open(source_txt, 'r') as f:
        source_dir = f.read().strip()
        
    f3x3_path = None
    best_f3x3 = None
    
    for root, dirs, files in os.walk(source_dir):
        if "f3x3.npy" in files:
            candidate_path = os.path.join(root, "f3x3.npy")
            f3x3_candidate = np.load(candidate_path)
            num_orig_points = f3x3_candidate.shape[0]
            
            if num_orig_points == len(df):
                f3x3_path = candidate_path
                best_f3x3 = f3x3_candidate
                break
                
            dropped_indices = [0, 25, 39]
            keep_mask = np.ones(num_orig_points, dtype=bool)
            if num_orig_points > max(dropped_indices):
                keep_mask[dropped_indices] = False
            
            identity_matrix = np.eye(3)
            is_zero_strain = np.max(np.max(np.abs(f3x3_candidate - identity_matrix), axis=2), axis=1) < 1e-6
            keep_mask[is_zero_strain] = False
            
            f3x3_kept = f3x3_candidate[keep_mask]
            
            if len(f3x3_kept) == len(df):
                f3x3_path = candidate_path
                best_f3x3 = f3x3_kept
                break
                
            if best_f3x3 is None:
                best_f3x3 = f3x3_kept

    if best_f3x3 is None:
        print(f"Error: f3x3.npy not found in {source_dir}")
        return
        
    f3x3_kept = best_f3x3
    
    if len(f3x3_kept) != len(df):
        print(f"Warning: Number of remaining f3x3 points ({len(f3x3_kept)}) does not match CSV rows ({len(df)})!")
        min_len = min(len(f3x3_kept), len(df))
        f3x3_kept = f3x3_kept[:min_len]
        df = df.iloc[:min_len]
        
    I1_bar, I2_bar, J = compute_invariants(f3x3_kept)
    I1_bar_m3 = I1_bar - 3.0
    I2_bar_m3 = I2_bar - 3.0
    J_m1_sq = (J - 1.0)**2
    
    if args.active_params:
        active_params = [p.strip() for p in args.active_params.split(',') if p.strip() in df.columns]
    else:
        active_params = []
        for param in params:
            if np.max(np.abs(df[param].values)) > 1e-4:
                active_params.append(param)
            
    num_params = len(active_params)
    if num_params == 0:
        print("No active parameters found with sensitivity > 1e-4.")
        return
        
    cols = 3
    rows = 1
    
    fig = plt.figure(figsize=(18, 6))
    
    pairs = [
        (I1_bar_m3, r"$\bar{I}_1 - 3$", I2_bar_m3, r"$\bar{I}_2 - 3$"),
        (I1_bar_m3, r"$\bar{I}_1 - 3$", J_m1_sq, r"$(J - 1)^2$"),
        (I2_bar_m3, r"$\bar{I}_2 - 3$", J_m1_sq, r"$(J - 1)^2$")
    ]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(active_params)))
    axes = []
    
    for c_idx, (x_data, x_label, y_data, y_label) in enumerate(pairs):
        ax = fig.add_subplot(rows, cols, c_idx + 1, projection='3d')
        axes.append(ax)
        
        for r_idx, param in enumerate(active_params):
            y_vals = df[param].values
            ax.scatter(x_data, y_data, y_vals, color=colors[r_idx], label=param, s=20, alpha=0.7)
            
        ax.set_title(f"Sensitivity vs {x_label} and {y_label}", fontsize=14, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
        ax.set_zlabel(r"$S_T$", fontsize=12, fontweight='bold')
        
        ax.grid(False)
        
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=len(active_params), fontsize=12, title="Parameters", title_fontsize=14)

    plt.tight_layout()    
    out_path = os.path.join(args.distilled_dir, f"{img_prefix}invariant_sensitivity_3d.png")
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {out_path}")

if __name__ == "__main__":
    main()
