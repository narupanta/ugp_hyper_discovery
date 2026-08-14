import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.material_models import get_material

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distilled_dir", type=str, required=True)
    parser.add_argument("--saved_model_dir", type=str, required=True)
    parser.add_argument("--material_model", type=str, default="gmr_nolog")
    args = parser.parse_args()
    
    out_dir = args.distilled_dir
    
    # Load samples
    samples_np = np.load(os.path.join(out_dir, "flow_samples.npy"))
    
    # Assume we know the parameter names for gmr_nolog
    full_param_names = ("C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03", "D1", "D2", "D3")
    df = pd.DataFrame(samples_np, columns=full_param_names)
    means = df.mean()
    
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
        
    model_folder_name = os.path.basename(os.path.normpath(args.saved_model_dir))
    parts = model_folder_name.split('_')
    true_model_name = parts[1] if len(parts) > 1 else "isihara"
    true_model = get_material(true_model_name, jit_P=False)
    
    true_params = {}
    if true_model_name == "isihara":
        true_params = {"C10": true_model.c10, "C01": true_model.c01, "C20": true_model.c20, "D1": true_model.d1}
    elif true_model_name in ["nh", "neohookean2", "nh2"]:
        true_params = {"C10": true_model.dev_params[0], "D1": true_model.vol_params[0]}
    elif true_model_name in ["gentthomas"]:
        true_params = {"C10": true_model.dev_params[0], "CL2": true_model.dev_params[10], "D1": true_model.vol_params[0]}
    
    true_vals = []
    pred_means = []
    pred_stds = []
    param_labels = []
    
    for col in full_param_names:
        if col in active_parameter_names:
            pred_means.append(means[col])
            pred_stds.append(df[col].std())
            true_vals.append(true_params.get(col, 0.0))
            param_labels.append(col)
            
    if len(pred_means) > 0:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.errorbar(true_vals, pred_means, yerr=pred_stds, fmt='o', color='#2980b9', capsize=5, label='Predicted vs True')
        
        min_val = min(min(true_vals), min(pred_means)) - 0.5
        max_val = max(max(true_vals), max(pred_means)) + 0.5
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='Parity (y=x)')
        
        for i, label in enumerate(param_labels):
            ax.annotate(label, (true_vals[i], pred_means[i]), xytext=(5, 5), textcoords='offset points')
            
        ax.set_xlabel('True Parameter Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Parameter Mean', fontsize=12, fontweight='bold')
        ax.set_title(f'Parameter Parity Plot ({args.material_model})', fontsize=14, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        
        parity_plot_path = os.path.join(out_dir, f"parameter_parity_{args.material_model}.pdf")
        plt.savefig(parity_plot_path, dpi=200)
        plt.close()
        print(f"Saved parameter parity plot to {parity_plot_path}")

if __name__ == "__main__":
    main()
