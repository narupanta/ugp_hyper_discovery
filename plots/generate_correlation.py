import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde
import os
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distilled_dir", type=str, required=True)
    parser.add_argument("--material_model", type=str, default="gmr_nolog")
    args = parser.parse_args()
    
    out_dir = args.distilled_dir
    
    # Load samples
    samples_np = np.load(os.path.join(out_dir, "flow_samples.npy"))
    
    if args.material_model in ["gmr", "gmr_log"]:
        full_param_names = ("C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03", "CL2", "D1", "D2", "D3")
    else:
        full_param_names = ("C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03", "D1", "D2", "D3")
    df = pd.DataFrame(samples_np, columns=full_param_names)
    
    # Load sensitivity CSV to find active parameters
    csv_dir = os.path.join(out_dir, "output", "sensitivities")
    if not os.path.exists(os.path.join(csv_dir, "total_sobol_indices_statistics_output_0.csv")):
        csv_dir = os.path.join(out_dir, "sensitivities")
    
    dfs = []
    # Could be up to 4 outputs
    for i in range(4):
        p = os.path.join(csv_dir, f"total_sobol_indices_statistics_output_{i}.csv")
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
            
    if dfs:
        param_cols = [c for c in dfs[0].columns if c not in ["Unnamed: 0", ""]]
        means_list = np.array([df_sub.iloc[0][param_cols].values.astype(float) for df_sub in dfs])
        mean_total_indices_outputs = np.nan_to_num(np.nanmean(means_list, axis=0), nan=0.0)
        relevant_parameter_indices = np.where(mean_total_indices_outputs >= 1e-4)[0].tolist()
        active_parameter_names = [full_param_names[i] for i in relevant_parameter_indices]
    else:
        active_parameter_names = full_param_names
        
    df_active = df[active_parameter_names]
    
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from core.material_models import get_material
    
    true_model_name = "isihara" # We can just hardcode or infer it
    parts = os.path.basename(os.path.normpath(args.distilled_dir)).split('_')
    for p in ["isihara", "nh", "neohookean2", "nh2", "gentthomas", "nh4", "neohookean4"]:
        if p in parts:
            true_model_name = p
            break
            
    true_model = get_material(true_model_name, jit_P=False)
    true_params = {}
    if true_model_name == "isihara":
        true_params = {"C10": true_model.c10, "C01": true_model.c01, "C20": true_model.c20, "D1": true_model.d1}
    elif true_model_name in ["nh", "neohookean2", "nh2"]:
        true_params = {"C10": true_model.dev_params[0], "D1": true_model.vol_params[0]}
    elif true_model_name in ["nh4", "neohookean4"]:
        true_params = {"C10": true_model.dev_params[0], "D2": true_model.vol_params[1]}
    elif true_model_name in ["gentthomas"]:
        true_params = {"C10": true_model.dev_params[0], "CL2": true_model.dev_params[10], "D1": true_model.vol_params[0]}
        
    if len(active_parameter_names) > 1:
        print(f"Generating parameter correlation pairplot for {len(active_parameter_names)} parameters using matplotlib...")
        # Use a smaller subset of samples to speed up plotting
        df_sample = df_active.sample(n=min(1000, len(df_active)), random_state=42)
        
        n_params = len(active_parameter_names)
        fig, axes = plt.subplots(n_params, n_params, figsize=(n_params*2.5, n_params*2.5))
        
        for i in range(n_params):
            for j in range(n_params):
                ax = axes[i, j]
                col_i = active_parameter_names[i]
                col_j = active_parameter_names[j]
                
                if i < j:
                    # Upper triangle - hide
                    ax.set_visible(False)
                elif i == j:
                    # Diagonal - Histogram
                    data = df_sample[col_i].values
                    ax.hist(data, bins=30, color='#16a085', alpha=0.7, density=True, edgecolor='white', linewidth=0.5)
                    
                    mean_val = data.mean()
                    true_val = true_params.get(col_i, 0.0)
                    ci_lower = np.percentile(data, 2.5)
                    ci_upper = np.percentile(data, 97.5)
                    
                    ax.axvline(mean_val, color='red', linestyle='-', lw=1.5, alpha=0.8)
                    ax.axvline(true_val, color='black', linestyle='--', lw=1.5, alpha=0.8)
                    ax.axvline(ci_lower, color='red', linestyle=':', lw=1.5, alpha=0.8)
                    ax.axvline(ci_upper, color='red', linestyle=':', lw=1.5, alpha=0.8)
                    
                    title_str = f"True: {true_val:.3f} | Mean: {mean_val:.3f}\n95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]"
                    ax.set_title(title_str, fontsize=9, fontweight='bold', pad=4)
                    
                    if i == n_params - 1:
                        ax.set_xlabel(col_j, fontsize=10, fontweight='bold')
                    else:
                        ax.set_xticklabels([])
                        
                    # Hide y-axis for diagonal
                    ax.set_yticks([])
                    if j == 0 and n_params > 1:
                        ax.set_ylabel(col_i, fontsize=10, fontweight='bold')
                else:
                    # Lower triangle - Scatter
                    ax.scatter(df_sample[col_j], df_sample[col_i], alpha=0.5, s=15, color='#2980b9', edgecolors='none')
                    
                    corr = df_sample[col_j].corr(df_sample[col_i])
                    ax.annotate(f"r = {corr:.3f}", xy=(0.95, 0.95), xycoords='axes fraction', 
                                ha='right', va='top', fontsize=10, fontweight='bold',
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
                                
                    if i == n_params - 1:
                        ax.set_xlabel(col_j, fontsize=10, fontweight='bold')
                    else:
                        ax.set_xticklabels([])
                    if j == 0:
                        ax.set_ylabel(col_i, fontsize=10, fontweight='bold')
                    else:
                        ax.set_yticklabels([])
                        
        fig.suptitle(f"Parameter Correlation Pairplot ({args.material_model})", y=1.02, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        pairplot_path = os.path.join(out_dir, f"parameter_correlation_{args.material_model}.png")
        fig.savefig(pairplot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved parameter correlation pairplot to {pairplot_path}")
    else:
        print("Not enough active parameters to generate a correlation pairplot.")

if __name__ == "__main__":
    main()
