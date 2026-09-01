import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compute_invariants(F):
    """Compute I1_bar, I2_bar, J for an array of deformation gradients."""
    # F shape: (N, 3, 3)
    J = np.linalg.det(F)
    
    # Compute C = F^T F
    F_T = np.transpose(F, (0, 2, 1))
    C = np.einsum('nij,njk->nik', F_T, F)
    
    # Isochoric part
    J_safe = np.clip(J, 1e-8, 1e8)
    C_bar = C / (J_safe**(2/3))[:, None, None]
    
    I1_bar = np.trace(C_bar, axis1=1, axis2=2)
    
    C_bar_sq = np.einsum('nij,njk->nik', C_bar, C_bar)
    I2_bar = 0.5 * (I1_bar**2 - np.trace(C_bar_sq, axis1=1, axis2=2))
    
    return I1_bar, I2_bar, J

def main():
    parser = argparse.ArgumentParser(description="Plot invariant-dependent Sobol indices")
    parser.add_argument("--distilled_dir", type=str, required=True, help="Path to the distilled model directory")
    parser.add_argument("--active_params", type=str, default=None, help="Comma-separated list of active parameters")
    parser.add_argument("--component", type=str, default="dev", choices=["dev", "vol", "aniso"], help="Component for split model")

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
    
    # Exclude non-parameter columns
    exclude_cols = ['test cases', 'Unnamed: 0']
    params = [c for c in df.columns if c not in exclude_cols and not c.startswith('Unnamed')]
    
    # Find the source extraction directory to load I_obs_all.npy
    source_txt = os.path.join(args.distilled_dir, source_txt_name)
    if not os.path.exists(source_txt):
        print(f"Error: {source_txt} not found. Cannot determine invariant values.")
        return
        
    with open(source_txt, 'r') as f:
        source_dir = f.read().strip()
        
    # Find f3x3.npy
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
                
            # Try applying mask
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
                best_f3x3 = f3x3_kept # Fallback

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
    
    if args.active_params:
        active_params = [p.strip() for p in args.active_params.split(',') if p.strip() in df.columns]
    else:
        # Filter active parameters (max sensitivity > 1e-4)
        active_params = []
        for param in params:
            if np.max(np.abs(df[param].values)) > 1e-4:
                active_params.append(param)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    invariants = [
        (r"$\bar{I}_1$", I1_bar),
        (r"$\bar{I}_2$", I2_bar),
        (r"$J$", J)
    ]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(active_params)))
    
    for i, (inv_name, inv_vals) in enumerate(invariants):
        ax = axes[i]
        
        for j, param in enumerate(active_params):
            y_vals = df[param].values
            ax.scatter(inv_vals, y_vals, label=param, alpha=0.7, s=20, color=colors[j])
            
        ax.set_ylim(bottom=-0.05, top=1.05)
        ax.set_title(f"Sensitivity vs {inv_name}", fontsize=14, fontweight='bold')
        ax.set_xlabel(inv_name, fontsize=12, fontweight='bold')
        ax.set_ylabel(r"Total-Order Sobol ($S_T$)", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    # Use a single legend for the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=len(active_params), fontsize=12, title="Parameters", title_fontsize=14)
    
    plt.tight_layout()    
    out_path = os.path.join(args.distilled_dir, f"{img_prefix}invariant_sensitivity_scatter.pdf")
    out_png = os.path.join(args.distilled_dir, f"{img_prefix}invariant_sensitivity_scatter.png")
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {out_path} and {out_png}")

if __name__ == "__main__":
    main()
